mod colour_shader;
mod process_shader;
mod vulkan_helper;

use crate::vulkan_helper::{execute_buffer, get_vulkan_device, make_compute_pipeline, make_image};
use image::{ImageBuffer, Rgba};
use noise::utils::{NoiseMapBuilder, PlaneMapBuilder};
use noise::{Fbm, Perlin};
use std::default::Default;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::{CopyBufferToImageInfo, CopyImageToBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, Image, ImageUsage};
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
        sun_pos: [-500.0f32, -500.0f32, 0.0f32],
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

    // Render loop
    println!("Starting render");
    let frame_count = 10;
    for frame_i in 0..frame_count {
        print!("Doing frame {} ", frame_i);

        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

        // render frame
        execute_buffer(render_command_buffer.clone(), queue.clone(), device.clone());

        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!("Took {:?}", end - start);

        // modify buffers
        {
            // save resulting image
            let buffer_data = out_buff.read().unwrap();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(IMAGE_WIDTH, IMAGE_HEIGHT, &buffer_data[..])
                    .unwrap();
            image.save(format!("frames/{}.png", frame_i)).unwrap();

            // move the sun
            program_data.sun_pos[2] += 0.5;
            let mut misc_data = misc_buffer.write().unwrap();
            *misc_data = program_data;
        }
    }

    println!("Success");
}
