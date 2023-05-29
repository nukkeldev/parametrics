use std::{borrow::Cow, mem};

use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix, Matrix4, Point3, SquareMatrix, Vector3};
use wgpu::util::DeviceExt;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::{surface_data as surface, transforms};

const ANIMATION_SPEED: f32 = 0.5;
const IS_PERSPECTIVE: bool = true;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Light {
    specular_color: [f32; 4],
    ambient_intensity: f32,
    diffuse_intensity: f32,
    specilar_intensity: f32,
    specilar_shininess: f32,
    is_two_side: i32,
}

pub fn light(sc: [f32; 3], ai: f32, di: f32, si: f32, ss: f32, its: i32) -> Light {
    Light {
        specular_color: [sc[0], sc[1], sc[2], 1.0],
        ambient_intensity: ai,
        diffuse_intensity: di,
        specilar_intensity: si,
        specilar_shininess: ss,
        is_two_side: its,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub color: [f32; 4],
}

pub fn vertex(p: [f32; 3], n: [f32; 3], c: [f32; 3]) -> Vertex {
    Vertex {
        position: [p[0], p[1], p[2], 1.0],
        normal: [n[0], n[1], n[2], 1.0],
        color: [c[0], c[1], c[2], 1.0],
    }
}

pub fn create_verticies(
    f: &dyn Fn(f32, f32) -> [f32; 3],
    colormap_name: &str,
    xmin: f32,
    xmax: f32,
    zmin: f32,
    zmax: f32,
    nx: usize,
    nz: usize,
    scale: f32,
    aspect: f32,
) -> Vec<Vertex> {
    let (pts, yrange) =
        surface::simple_surface_points(f, xmin, xmax, zmin, zmax, nx, nz, scale, aspect);
    let pos = surface::simple_surface_positions(&pts, nx, nz);
    let normal = surface::simple_surface_normals(&pts, nx, nz);
    let color = surface::simple_surface_colors(&pts, nx, nz, yrange, colormap_name);
    (0..pos.len())
        .map(|i| vertex(pos[i], normal[i], color[i]))
        .collect()
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

struct State {
    pub init: transforms::InitWgpu,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_uniform_buffer: wgpu::Buffer,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    num_vertices: u32,

    mouse_pressed: bool,
    mouse_delta: (f32, f32),
    zoom: f32,

    camera_pos: Point3<f32>,
}

impl State {
    async fn new(
        window: &Window,
        pos_data: &Vec<[f32; 3]>,
        normal_data: &Vec<[f32; 3]>,
        color_data: &Vec<[f32; 3]>,
        index_data: &Vec<u32>,
        light_data: Light,
    ) -> Self {
        let vertex_data: Vec<Vertex> = (0..pos_data.len())
            .map(|i| vertex(pos_data[i], normal_data[i], color_data[i]))
            .collect();

        let init = transforms::InitWgpu::init_wgpu(window).await;

        let shader = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

        // Uniform Data
        let camera_pos = (3.0, 3.0, 3.0).into();
        let look_dir = (0.0, 0.0, 0.0).into();
        let up_dir = cgmath::Vector3::unit_y();

        let (view_mat, project_mat, _) = transforms::create_view_projection(
            camera_pos,
            look_dir,
            up_dir,
            init.config.width as f32 / init.config.height as f32,
            IS_PERSPECTIVE,
        );

        // Create the `vertex_uniform_buffer`
        // `model_mat` and `view_projection_mat` will be stored in the `vertex_unifrom_buffer` inside the update function
        // Rather than declaring them seperately as fields of `State`
        let vertex_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Uniform Buffer"),
            size: 192, // 3 * mat4x4 (64 bytes)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create the `fragment_uniform_buffer`
        let fragment_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fragment Uniform Buffer"),
            size: 32, // 2 * vec4
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Store the light and eye positions
        // These will not change with the rotation animation so they can be written upon initializion
        let light_position: &[f32; 3] = camera_pos.as_ref();
        let eye_position: &[f32; 3] = camera_pos.as_ref();
        init.queue.write_buffer(
            &fragment_uniform_buffer,
            0,
            bytemuck::cast_slice(light_position),
        );
        init.queue.write_buffer(
            &fragment_uniform_buffer,
            16, // size of light_position
            bytemuck::cast_slice(eye_position),
        );

        // Create the `light_uniform_buffer`
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 48, // 2 * vec4 + 4 * f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Store the light parameters
        init.queue.write_buffer(
            &light_uniform_buffer,
            0,
            bytemuck::cast_slice(&[light_data]),
        );

        let uniform_bind_group_layout =
            init.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Bind Group Layout"),
                    entries: &[
                        // Vertex Buffer Layout
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Fragment Buffer Layout
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Light Buffer Layout
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let uniform_bind_group = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[
                // Vertex Buffer Binding
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_uniform_buffer.as_entire_binding(),
                },
                // Fragment Buffer Binding
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fragment_uniform_buffer.as_entire_binding(),
                },
                // Light Buffer Binding
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = init
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: init.config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                // Tells `wgpu` when to draw over a pixel and when not to
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let vertex_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

        let num_vertices = index_data.len() as u32;

        Self {
            init,
            pipeline,
            vertex_buffer,
            index_buffer,
            vertex_uniform_buffer,
            uniform_bind_group,
            view_mat,
            project_mat,
            num_vertices,
            mouse_pressed: false,
            mouse_delta: (0.0, 0.0),
            zoom: 1.0,
            camera_pos: Point3::new(3.0, 3.0, 3.0),
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat = transforms::create_projection(
                new_size.width as f32 / new_size.height as f32,
                IS_PERSPECTIVE,
            );
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    self.mouse_delta.0 += delta.0 as f32;
                    self.mouse_delta.1 += delta.1 as f32;
                }
                true
            }
            DeviceEvent::MouseWheel { delta } => {
                if let MouseScrollDelta::PixelDelta(pos) = delta {
                    self.zoom += pos.y as f32;
                }
                true
            }
            DeviceEvent::Button { button: 1, state } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    const LOOK_DIR: Point3<f32> = Point3::new(0.0, 0.0, 0.0);
    const UP_DIR: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);

    fn update(&mut self, dt: std::time::Duration) {
        let dt = ANIMATION_SPEED * dt.as_secs_f32();

        self.camera_pos.x += self.mouse_delta.0;
        self.camera_pos.y += self.mouse_delta.1;
        self.mouse_delta = (0.0, 0.0);

        let view_mat = transforms::create_view(self.camera_pos, Self::LOOK_DIR, Self::UP_DIR);

        let model_mat =
            transforms::create_transforms([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Apparent the order fucking matters when doing matrix multiplication bruh
        let view_project_mat = self.project_mat * view_mat;

        let normal_mat = model_mat.invert().unwrap().transpose();

        let model_ref: &[f32; 16] = model_mat.as_ref();
        let view_projection_ref: &[f32; 16] = view_project_mat.as_ref();
        let normal_ref: &[f32; 16] = normal_mat.as_ref();

        self.init.queue.write_buffer(
            &self.vertex_uniform_buffer,
            0,
            bytemuck::cast_slice(model_ref),
        );
        self.init.queue.write_buffer(
            &self.vertex_uniform_buffer,
            64,
            bytemuck::cast_slice(view_projection_ref),
        );
        self.init.queue.write_buffer(
            &self.vertex_uniform_buffer,
            128,
            bytemuck::cast_slice(normal_ref),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.init.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.init.config.width,
                height: self.init.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[wgpu::TextureFormat::Depth24Plus],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_vertices, 0, 0..1);
        }

        self.init.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

const TARGET_FPS: u64 = 60;

pub fn run(
    pos_data: &Vec<[f32; 3]>,
    normal_data: &Vec<[f32; 3]>,
    color_data: &Vec<[f32; 3]>,
    index_data: &Vec<u32>,
    light_data: Light,
) {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    window.set_title("Parametric 3D Surface");

    let mut state = pollster::block_on(State::new(
        &window,
        &pos_data,
        &normal_data,
        &color_data,
        &index_data,
        light_data,
    ));
    let render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let start_time = std::time::Instant::now();
        match event {
            Event::DeviceEvent { ref event, .. } => {
                state.input(event);
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - render_start_time;
                state.update(dt);

                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(wgpu::SurfaceError::Outdated) => {}
                    Err(e) => eprintln!("Error: {e:#?}"),
                }
            }
            Event::MainEventsCleared => window.request_redraw(),
            _ => {}
        }

        if *control_flow == ControlFlow::Exit {
            return;
        }

        let elapsed_time = std::time::Instant::now()
            .duration_since(start_time)
            .as_millis() as u64;
        let wait_millis = match 1000 / TARGET_FPS >= elapsed_time {
            true => 1000 / TARGET_FPS - elapsed_time,
            false => 0,
        };
        let new_inst = start_time + std::time::Duration::from_millis(wait_millis);
        *control_flow = ControlFlow::WaitUntil(new_inst);
    });
}
