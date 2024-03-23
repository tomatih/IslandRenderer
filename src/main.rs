use std::default::Default;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use cgmath::{Vector3, Zero};
use cgmath::VectorSpace;
use noise::{Fbm, Perlin};
use noise::utils::{NoiseMapBuilder, PlaneMapBuilder};
use video_rs::{Encoder, Time};
use video_rs::encode::Settings;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{Pipeline, PipelineBindPoint},
};
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{CopyBufferToImageInfo, CopyImageToBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage, view::ImageView};

use crate::vulkan_helper::{execute_buffer, get_vulkan_device, make_compute_pipeline, make_image};

mod colour_shader;
mod process_shader;
mod vulkan_helper;

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
struct ProgramData {
    sun_pos: [f32; 3],
}

const IMAGE_WIDTH: u32 = 1024;
const IMAGE_HEIGHT: u32 = 1024;

fn setup(
    device: Arc<Device>,
    queue: Arc<Queue>,
    noise_image: Arc<Image>,
    noise_image_view: Arc<ImageView>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_allocator: &StandardDescriptorSetAllocator,
    command_buffer_allocator: &StandardCommandBufferAllocator,
) {
    // init noise generation
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let fbm = Fbm::<Perlin>::new(seed.as_secs() as u32);
    let noise_map = PlaneMapBuilder::<_, 2>::new(&fbm)
        .set_size(IMAGE_WIDTH as usize, IMAGE_HEIGHT as usize)
        .set_x_bounds(-5.0, 5.0)
        .set_y_bounds(-5.0, 5.0)
        .build();

    // noise buffer
    let in_buff = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        noise_map.iter().map(|x| ((x + 1.0) * 0.5 * 255.0) as u8),
    )
    .unwrap();

    //setup pipeline
    let setup_shader = process_shader::load(device.clone()).unwrap();
    let setup_pipeline = make_compute_pipeline(setup_shader, device.clone());

    // setup descriptor sets
    let setup_layout = setup_pipeline.layout().set_layouts().first().unwrap();
    let setup_set = PersistentDescriptorSet::new(
        descriptor_allocator,
        setup_layout.clone(),
        [WriteDescriptorSet::image_view(0, noise_image_view.clone())],
        [],
    )
    .unwrap();

    // make command buffer
    let mut setup_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    setup_builder
        .bind_pipeline_compute(setup_pipeline.clone())
        .unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            in_buff.clone(),
            noise_image.clone(),
        ))
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            setup_pipeline.layout().clone(),
            0,
            setup_set,
        )
        .unwrap()
        .dispatch([IMAGE_WIDTH / 8, IMAGE_HEIGHT / 8, 1])
        .unwrap();

    let setup_command_buffer = setup_builder.build().unwrap();

    // perform setup
    execute_buffer(setup_command_buffer, queue, device);
}

fn update_data(data: &ProgramData, buffer: &Subbuffer<ProgramData>){
    let mut buff_writer = buffer.write().unwrap();
    *buff_writer = *data;
}

fn main() {
    println!("Setting up vulkan");
    // Initialize vulkan
    let (device, queue) = get_vulkan_device();

    // Prepare memory allocators
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // Create buffers
    let mut program_data = ProgramData {
        sun_pos: Vector3::zero().into(),
    };
    let misc_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        program_data,
    )
    .unwrap();

    let out_buff = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..IMAGE_WIDTH * IMAGE_HEIGHT * 4).map(|_| 0u8),
    )
    .expect("failed to create output buffer");

    // Create GPU images
    let in_image = make_image(
        Format::R8_UNORM,
        [IMAGE_WIDTH, IMAGE_HEIGHT, 1],
        ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
        memory_allocator.clone(),
    );
    let in_image_view = ImageView::new_default(in_image.clone()).unwrap();

    let out_image = make_image(
        Format::R8G8B8A8_UNORM,
        [IMAGE_WIDTH, IMAGE_HEIGHT, 1],
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
        memory_allocator.clone(),
    );
    let out_image_view = ImageView::new_default(out_image.clone()).unwrap();

    // Create the compute pipeline
    let render_shader = colour_shader::load(device.clone()).unwrap();
    let render_pipeline = make_compute_pipeline(render_shader, device.clone());

    // create descriptor set
    let layout = render_pipeline.layout().set_layouts().first().unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, out_image_view.clone()),
            WriteDescriptorSet::image_view(1, in_image_view.clone()),
            WriteDescriptorSet::buffer(2, misc_buffer.clone()),
        ],
        [],
    )
    .unwrap();

    // In order to execute our operation, we have to build a command buffer.
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();
    builder
        .bind_pipeline_compute(render_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            render_pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .dispatch([IMAGE_WIDTH / 8, IMAGE_HEIGHT / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            out_image.clone(),
            out_buff.clone(),
        ))
        .unwrap();

    // Finish building the command buffer by calling `build`.
    let render_command_buffer = builder.build().unwrap();

    // run setup
    println!("Doing pre-processing pass");
    setup(
        device.clone(),
        queue.clone(),
        in_image.clone(),
        in_image_view.clone(),
        memory_allocator.clone(),
        &descriptor_set_allocator,
        &command_buffer_allocator,
    );

    // Render data
    println!("Starting render");
    let frame_count = 30;
    let sun_start = Vector3::new(-1024.0f32, -1024.0f32, 0.0f32);
    let sun_apex = Vector3::new(-1024.0f32, 0.0f32, 5.0f32);
    let sun_stop = Vector3::new(-1024.0f32, 1024.0f32, 0.0f32);

    // video encoder
    video_rs::init().unwrap();
    let video_settings = Settings::preset_h264_yuv420p(IMAGE_WIDTH as usize, IMAGE_HEIGHT as usize, false);
    let mut encoder = Encoder::new(Path::new("out.mp4"), video_settings).expect("Field to create video encoder");
    let frame_duration = Time::from_nth_of_a_second(15);
    let mut frame_position = Time::zero();

    // render loop
    program_data.sun_pos = sun_start.into();
    update_data(&program_data, &misc_buffer);
    for frame_i in 0..=frame_count {
        print!("Doing frame {} ", frame_i);

        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

        // render frame
        execute_buffer(render_command_buffer.clone(), queue.clone(), device.clone());

        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!("Took {:?}", end - start);

        // modify buffers
        {
            // save resulting frame
            let buffer_data = out_buff.read().unwrap();
            let frame = ndarray::Array3::from_shape_fn(
                (IMAGE_HEIGHT as usize, IMAGE_WIDTH as usize, 3usize),
                |(y,x,c)| buffer_data[(y*IMAGE_WIDTH as usize+x)*4+c]  // RGBA -> RGB
            );
            encoder.encode(&frame, &frame_position).expect("Failed encoding a frame");
            frame_position = frame_position.aligned_with(&frame_duration).add();

            // move the sun
            let mut amount = frame_i as f32/frame_count as f32 * 2.0;
            program_data.sun_pos = if frame_i < frame_count/2 {
                Vector3::lerp(sun_start, sun_apex, amount)
            }else {
                amount-=1.0;
                Vector3::lerp(sun_apex, sun_stop, amount)
            }.into();

            update_data(&program_data, &misc_buffer);
        }
    }

    encoder.finish().expect("failed to finish encoder");

    println!("Success");
}
