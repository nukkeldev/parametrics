#![allow(dead_code)]

use cgmath::*;
use std::f32::consts::PI;
use winit::window::Window;

pub struct InitWgpu {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl InitWgpu {
    pub async fn init_wgpu(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter.");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device.");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            present_mode: wgpu::PresentMode::Mailbox,
            ..surface
                .get_default_config(&adapter, size.width, size.height)
                .unwrap()
        };

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }
}

/// `wgpu` is based on DirectX and Metal's coordinate system where the Normalized Device Coordinate (NDL) ranges from \[-1, 1\] for x and y and from \[0, 1\] for z.
/// Whereas `cgmath` is based on OpenGL which has an NDL of \[-1, 1\] for x, y, and z.
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0
);

pub fn create_view_projection(
    camera_pos: Point3<f32>,
    look_dir: Point3<f32>,
    up_dir: Vector3<f32>,
    aspect: f32,
    is_perspective: bool,
) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    // Construct the view matrix
    // `look_at_rh` uses a right-handed coordinate system and `look_at_lh` does the opposite
    let view_mat = create_view(camera_pos, look_dir, up_dir);

    // Construct the projection matrix
    let project_mat = create_projection(aspect, is_perspective);

    // Const the view-projection matrix
    let view_project_mat = view_mat * project_mat;

    // Return the various matrices
    (view_mat, project_mat, view_project_mat)
}

pub fn create_view(
    camera_pos: Point3<f32>,
    look_dir: Point3<f32>,
    up_dir: Vector3<f32>,
) -> Matrix4<f32> {
    Matrix4::look_at_rh(camera_pos, look_dir, up_dir)
}

pub fn create_projection(aspect: f32, is_perspective: bool) -> Matrix4<f32> {
    if is_perspective {
        OPENGL_TO_WGPU_MATRIX * perspective(Rad(2.0 * PI / 5.0), aspect, 0.1, 100.0)
    } else {
        OPENGL_TO_WGPU_MATRIX * ortho(-4.0, 4.0, -3.0, 3.0, -1.0, 6.0)
    }
}

pub fn create_transforms(
    translation: [f32; 3],
    rotation: [f32; 3],
    scaling: [f32; 3],
) -> Matrix4<f32> {
    // Create the transfromation matrices
    let trans_mat =
        Matrix4::from_translation(Vector3::new(translation[0], translation[1], translation[2]));
    let rotate_mat_x = Matrix4::from_angle_x(Rad(rotation[0]));
    let rotate_mat_y = Matrix4::from_angle_y(Rad(rotation[1]));
    let rotate_mat_z = Matrix4::from_angle_z(Rad(rotation[2]));
    let scale_mat = Matrix4::from_nonuniform_scale(scaling[0], scaling[1], scaling[2]);

    // Combine all of the transfromation matrices together to format a final transformation matrix
    // Construct and return the model matrix
    trans_mat * rotate_mat_x * rotate_mat_y * rotate_mat_z * scale_mat
}
