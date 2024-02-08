mod procees_shader;
mod colour_shader;

use std::default::Default;
use std::fs::File;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use image::{ImageBuffer, Rgba};
use noise::{Fbm, Perlin};
use noise::utils::{NoiseMapBuilder, PlaneMapBuilder};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
    VulkanLibrary,
};
use vulkano::command_buffer::{CopyBufferToImageInfo, CopyImageToBufferInfo};
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView};

fn main() {
    // As with other examples, the first step is to create an instance.
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
        .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
        .unwrap();

    // Get a queue
    let queue = queues.next().unwrap();

    // Create the compute pipeline
    let pipeline = {
        let cs = colour_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
            .unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
            .unwrap()
    };

    let setup_pipeline = {
        let cs = procees_shader::load(device.clone()).unwrap().entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout)
        ).unwrap()
    };

    // Prepare memory
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let misc_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo{
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [-500.0f32, -500.0f32, 0.0f32]
    ).unwrap();

    // init noise generation
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let fbm = Fbm::<Perlin>::new(seed.as_secs() as u32);
    let noise_map = PlaneMapBuilder::<_,2>::new(&fbm)
        .set_size(1024, 1024)
        .set_x_bounds(-5.0,5.0)
        .set_y_bounds(-5.0, 5.0)
        .build();
    // create input buffer
    let in_buff = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo{
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS ,
            ..Default::default()
        },
        noise_map.iter().map(|x| ((x+1.0)*0.5*255.0) as u8 )
    ).unwrap();

    // Create GPU image
    let out_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).unwrap();
    let out_image_view = ImageView::new_default(out_image.clone()).unwrap();

    let in_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo{
            image_type: ImageType::Dim2d,
            format: Format::R8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    ).unwrap();
    let in_image_view = ImageView::new_default(in_image.clone()).unwrap();

    // create descriptor set
    let layout = pipeline.layout().set_layouts().first().unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, out_image_view.clone()),
            WriteDescriptorSet::image_view(1, in_image_view.clone()),
            WriteDescriptorSet::buffer(2,misc_buffer.clone())
        ],
        [],
    ).unwrap();

    let setup_layout = setup_pipeline.layout().set_layouts().first().unwrap();
    let setup_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        setup_layout.clone(),
        [
            WriteDescriptorSet::image_view(0, in_image_view.clone())
        ],
        []
    ).unwrap();

    // crete output buffer
    let out_buff = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    ).expect("failed to create output buffer");


    let mut setup_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();

    setup_builder
        .bind_pipeline_compute(setup_pipeline.clone()).unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            in_buff.clone(),
            in_image.clone()
        )).unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            setup_pipeline.layout().clone(),
            0,
            setup_set,
        ).unwrap()
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap();

    let setup_command_buffer = setup_builder.build().unwrap();

    // In order to execute our operation, we have to build a command buffer.
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
        .unwrap();
    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            out_image.clone(),
            out_buff.clone(),
        ))
        .unwrap();

    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();


    sync::now(device.clone())
        .then_execute(queue.clone(), setup_command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();


    // Render loop
    println!("Starting render");
    let frame_count = 10;
    let mut last_sun_pos = 0.0f32;
    for frame_i in 0..frame_count{
        println!("Doing frame {}",frame_i);
        // render frame
        sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // modify buffers
        {
            // save resulting image
            let buffer_data = out_buff.read().unwrap();
            let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_data[..]).unwrap();
            image.save(format!("frames/{}.png",frame_i)).unwrap();

            // move the sun
            last_sun_pos += 0.5;
            let mut misc_data = misc_buffer.write().unwrap();
            misc_data[2] = last_sun_pos;
        }
    }

    println!("Success");
}
