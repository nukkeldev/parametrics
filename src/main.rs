use std::f32::consts::PI;

mod colormap;
mod common;
mod math_func;
mod surface_data;
mod transforms;
mod vertex_data;

macro_rules! parametric_decl {
    ($name:ident, u: $umin:expr => $umax:expr, v: $vmin:expr => $vmax:expr$(, $prop:ident = $val:expr)*) => {
        surface_data::ParametricSurface {
            f: math_func::$name,
            umin: $umin,
            umax: $umax,
            vmin: $vmin,
            vmax: $vmax,
            $(
                $prop: $val,
            )*
            ..Default::default()
        }
    };
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let function_selection = args.next().unwrap_or("".to_string());

    let ps_struct = match function_selection.as_str() {
        "klein" => parametric_decl!(klein_bottle, u: 0.0 => PI, v: 0.0 => (2.0 * PI)),
        "wellenkugel" => surface_data::ParametricSurface {
            f: math_func::wellenkugel,
            umin: 0.0,
            umax: 14.5,
            vmin: 0.0,
            vmax: 5.0, // 2PI = full circumference
            u_segments: 100,
            v_segments: 50,
            scale: 0.17,
            colormap_name: "cool",
            ..Default::default()
        },
        "sphere" => surface_data::ParametricSurface {
            f: math_func::sphere,
            umin: 0.0,
            umax: 2.0 * PI,
            vmin: 0.0,
            vmax: PI,
            u_segments: 100,
            v_segments: 50,
            scale: 1.0,
            params: [1.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        },
        "hyperbola" => surface_data::ParametricSurface {
            f: math_func::hyperbola,
            umin: -2.0,
            umax: 2.0,
            vmin: 0.0,
            vmax: 2.0 * PI,
            u_segments: 100,
            v_segments: 50,
            scale: 0.25,
            params: [1.0, 1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        },
        "plane" => surface_data::ParametricSurface {
            f: math_func::plane,
            umin: -2.0,
            umax: 2.0,
            vmin: -2.0,
            vmax: 2.0,
            u_segments: 100,
            v_segments: 100,
            scale: 0.5,
            // [0] = Height displacement
            params: [0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        },
        "cone" => surface_data::ParametricSurface {
            f: math_func::cone,
            umin: 0.0,
            umax: 2.0 * PI,
            vmin: 0.0,
            vmax: 1.0,
            u_segments: 100,
            v_segments: 100,
            scale: 1.0,
            // [0] = radius of the base, [1] = height
            params: [1.0, 2.0, 0.0, 0.0, 0.0],
            ..Default::default()
        },
        "mobius-strip" => surface_data::ParametricSurface {
            f: math_func::mobius_strip,
            umin: 0.0,
            umax: 2.0 * PI,
            vmin: -1.0,
            vmax: 1.0,
            u_segments: 100,
            v_segments: 100,
            scale: 0.75,
            // [0] = radius of the central circle
            params: [2.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        },
        "ellipsoid" => surface_data::ParametricSurface {
            f: math_func::ellipsoid,
            umin: 0.0,
            umax: 2.0 * PI,
            vmin: -0.01, // Odd, should be 0, but then theres a hole
            vmax: PI,
            u_segments: 100,
            v_segments: 100,
            scale: 1.0,
            // [0, 1, 2] = semi-axes of x, y, z
            params: [2.0, 1.0, 1.0, 0.0, 0.0],
            ..Default::default()
        },
        "cylinder" => {
            parametric_decl!(cylinder, u: 0.0 => (2.0 * PI), v: 0.0 => PI, scale = 0.5, params = [1.0, 0.0, 0.0, 0.0, 0.0])
        }
        _ => surface_data::ParametricSurface::default(),
    };

    let (pos_data, normal_data, color_data, index_data) = ps_struct.new();
    let light_data = common::light([1.0, 1.0, 1.0], 0.1, 0.8, 0.4, 30.0, 1);

    common::run(
        &pos_data,
        &normal_data,
        &color_data,
        &index_data,
        light_data,
    );
}
