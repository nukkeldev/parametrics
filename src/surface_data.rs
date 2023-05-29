#![allow(dead_code)]
use std::f32::consts::PI;

use cgmath::*;

use crate::{colormap, math_func};

pub struct ParametricSurface {
    pub f: fn(f32, f32, [f32; 5]) -> [f32; 3],
    pub umin: f32,
    pub umax: f32,
    pub vmin: f32,
    pub vmax: f32,
    pub u_segments: usize,
    pub v_segments: usize,
    pub scale: f32,
    pub aspect: f32,
    pub use_colormap: bool,
    pub colormap_name: &'static str,
    pub colormap_direction: &'static str,
    pub color: [f32; 3],
    pub params: [f32; 5],
}

impl Default for ParametricSurface {
    fn default() -> Self {
        Self {
            f: math_func::torus,
            umin: 0.0,
            umax: 2.0 * PI,
            vmin: 0.0,
            vmax: 2.0 * PI,
            u_segments: 100,
            v_segments: 100,
            scale: 1.5,
            aspect: 1.0,
            use_colormap: true,
            colormap_name: "jet",
            colormap_direction: "y",
            color: [1.0, 0.0, 0.0],
            params: [1.0, 0.3, 0.0, 0.0, 0.0],
        }
    }
}

impl ParametricSurface {
    pub fn new(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        let Self {
            f,
            umin,
            umax,
            vmin,
            vmax,
            u_segments,
            v_segments,
            scale,
            aspect,
            use_colormap,
            colormap_name,
            colormap_direction,
            color,
            params,
        } = *self;

        let n_vertices = (u_segments + 1) * (v_segments + 1);
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
        let mut colors: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);

        let du = (umax - umin) / u_segments as f32;
        let dv = (vmax - vmin) / v_segments as f32;

        let eps = 1e-5;
        let mut p0: Vector3<f32>;
        let mut p1: Vector3<f32>;
        let mut p2: Vector3<f32>;
        let mut p3: Vector3<f32>;
        let mut pa: [f32; 3];

        let cd = match colormap_direction {
            "x" => 0,
            "z" => 2,
            _ => 1,
        };

        let (min, max) = Self::parametric_surface_range(&self, cd);

        let vertex =
            |pa: [f32; 3]| Vector3::new(pa[0] * scale, scale * aspect * pa[1], scale * pa[2]);

        for i in 0..=u_segments {
            let u = umin + i as f32 * du;
            for j in 0..=v_segments {
                let v = vmin + j as f32 * dv;
                pa = f(u, v, params);
                p0 = vertex(pa);

                // Calculate Normals
                if u - eps >= 0.0 {
                    pa = f(u - eps, v, params);
                    p1 = vertex(pa);
                    p2 = p0 - p1;
                } else {
                    pa = f(u + eps, v, params);
                    p1 = vertex(pa);
                    p2 = p1 - p0;
                }

                if v - eps >= 0.0 {
                    pa = f(u, v - eps, params);
                    p1 = vertex(pa);
                    p3 = p0 - p1;
                } else {
                    pa = f(u, v + eps, params);
                    p1 = vertex(pa);
                    p3 = p1 - p0;
                }
                let normal = p3.cross(p2).normalize();

                // Calculate the colormap
                let vertex_color = if use_colormap {
                    colormap::color_interp(colormap_name, min, max, p0[cd])
                } else {
                    color
                };

                positions.push(p0.into());
                normals.push(normal.into());
                colors.push(vertex_color);
            }
        }

        let n_face = u_segments * v_segments;
        let n_triangles = n_face * 2;
        let n_indices = n_triangles * 3;

        let mut indices: Vec<u32> = Vec::with_capacity(n_indices);

        let n_vertices_per_row = v_segments + 1;

        for i in 0..u_segments {
            for j in 0..v_segments {
                let idx0 = j + i * n_vertices_per_row;
                let idx1 = j + 1 + i * n_vertices_per_row;
                let idx2 = j + 1 + (i + 1) * n_vertices_per_row;
                let idx3 = j + (i + 1) * n_vertices_per_row;

                indices.push(idx0 as u32);
                indices.push(idx1 as u32);
                indices.push(idx2 as u32);

                indices.push(idx2 as u32);
                indices.push(idx3 as u32);
                indices.push(idx0 as u32);
            }
        }

        (positions, normals, colors, indices)
    }

    fn parametric_surface_range(&self, dir: usize) -> (f32, f32) {
        let Self {
            f,
            umin,
            umax,
            vmin,
            vmax,
            u_segments,
            v_segments,
            scale,
            aspect,
            params,
            ..
        } = *self;

        let du = (umax - umin) / u_segments as f32;
        let dv = (vmax - vmin) / v_segments as f32;

        let mut min = std::f32::MAX;
        let mut max = std::f32::MIN;

        for i in 0..u_segments {
            let u = umin + i as f32 * du;
            for j in 0..v_segments {
                let v = vmin + j as f32 * dv;
                let mut pt = f(u, v, params);
                pt = [pt[0] * scale, scale * aspect * pt[1], scale * pt[2]];
                min = min.min(pt[dir]);
                max = max.max(pt[dir]);
            }
        }

        (min, max)
    }
}

pub fn simple_surface_positions(pts: &Vec<Vec<[f32; 3]>>, nx: usize, nz: usize) -> Vec<[f32; 3]> {
    let mut positions = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    for i in 0..nx - 1 {
        for j in 0..nz - 1 {
            let p0 = pts[i][j];
            let p1 = pts[i][j + 1];
            let p2 = pts[i + 1][j + 1];
            let p3 = pts[i + 1][j];

            positions.push(p0);
            positions.push(p1);
            positions.push(p2);
            positions.push(p2);
            positions.push(p3);
            positions.push(p0);
        }
    }
    positions
}

pub fn simple_surface_normals(pts: &Vec<Vec<[f32; 3]>>, nx: usize, nz: usize) -> Vec<[f32; 3]> {
    let mut normals = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    for i in 0..nx - 1 {
        for j in 0..nz - 1 {
            let p0 = pts[i][j];
            let p1 = pts[i][j + 1];
            let p2 = pts[i + 1][j + 1];
            let p3 = pts[i + 1][j];

            // Normals
            let ca = Vector3::new(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
            let db = Vector3::new(p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]);
            let cp = ca.cross(db).normalize();

            for _ in 0..6 {
                normals.push([cp[0], cp[1], cp[2]]);
            }
        }
    }
    normals
}

pub fn simple_surface_colors(
    pts: &Vec<Vec<[f32; 3]>>,
    nx: usize,
    nz: usize,
    yrange: [f32; 2],
    colormap_name: &str,
) -> Vec<[f32; 3]> {
    let mut colors: Vec<[f32; 3]> = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    for i in 0..nx - 1 {
        for j in 0..nz - 1 {
            let p0 = pts[i][j];
            let p1 = pts[i][j + 1];
            let p2 = pts[i + 1][j + 1];
            let p3 = pts[i + 1][j];

            let c0 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p0[1]);
            let c1 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p1[1]);
            let c2 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p2[1]);
            let c3 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p3[1]);

            colors.push(c0);
            colors.push(c1);
            colors.push(c2);
            colors.push(c2);
            colors.push(c3);
            colors.push(c0);
        }
    }
    colors
}

pub fn simple_surface_points(
    f: &dyn Fn(f32, f32) -> [f32; 3],
    xmin: f32,
    xmax: f32,
    zmin: f32,
    zmax: f32,
    nx: usize,
    nz: usize,
    scale: f32,
    aspect: f32,
) -> (Vec<Vec<[f32; 3]>>, [f32; 2]) {
    // Scale of each cell
    let dx = (xmax - xmin) / ((nx - 1) as f32);
    let dz = (zmax - zmin) / ((nz - 1) as f32);

    let mut ymin: f32 = 0.0;
    let mut ymax: f32 = 0.0;

    let mut pts: Vec<Vec<[f32; 3]>> = vec![vec![Default::default(); nz]; nx];
    for i in 0..nx {
        let x = xmin + i as f32 * dx;
        let pt1: Vec<[f32; 3]> = (0..nz)
            .map(|j| {
                let z = zmin + j as f32 * dz;
                let pt = f(x, z);

                ymin = if pt[1] < ymin { pt[1] } else { ymin };
                ymax = if pt[1] > ymax { pt[1] } else { ymax };

                pt
            })
            .collect();
        pts[i] = pt1;
    }

    let ymin1 = ymin - (1.0 - aspect) * (ymax - ymin);
    let ymax1 = ymax + (1.0 - aspect) * (ymax - ymin);

    for i in 0..nx {
        for j in 0..nz {
            pts[i][j] = normalize_point(pts[i][j], xmin, xmax, ymin1, ymax1, zmin, zmax, scale);
        }
    }

    let cmin = normalize_point(
        [0.0, ymin, 0.0],
        xmin,
        xmax,
        ymin1,
        ymax1,
        zmin,
        zmax,
        scale,
    )[1];
    let cmax = normalize_point(
        [0.0, ymax, 0.0],
        xmin,
        xmax,
        ymin1,
        ymax1,
        zmin,
        zmax,
        scale,
    )[1];

    return (pts, [cmin, cmax]);
}

fn normalize_point(
    pt: [f32; 3],
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    zmin: f32,
    zmax: f32,
    scale: f32,
) -> [f32; 3] {
    let px = scale * (-1.0 + 2.0 * (pt[0] - xmin) / (xmax - xmin));
    let py = scale * (-1.0 + 2.0 * (pt[1] - ymin) / (ymax - ymin));
    let pz = scale * (-1.0 + 2.0 * (pt[2] - zmin) / (zmax - zmin));
    [px, py, pz]
}

pub fn simple_surface(
    pts: &Vec<Vec<[f32; 3]>>,
    nx: usize,
    nz: usize,
    yrange: [f32; 2],
    colormap_name: &str,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut positions = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    let mut normals = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    let mut colors = Vec::with_capacity((4 * (nx - 1) * (nz - 1)) as usize);
    for i in 0..nx - 1 {
        for j in 0..nz - 1 {
            let p0 = pts[i][j];
            let p1 = pts[i][j + 1];
            let p2 = pts[i + 1][j + 1];
            let p3 = pts[i + 1][j];

            positions.push(p0);
            positions.push(p1);
            positions.push(p2);
            positions.push(p2);
            positions.push(p3);
            positions.push(p0);

            // Normals
            let ca = Vector3::new(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
            let db = Vector3::new(p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]);
            let cp = ca.cross(db).normalize();

            for _ in 0..6 {
                normals.push([cp[0], cp[1], cp[2]]);
            }

            let c0 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p0[1]);
            let c1 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p1[1]);
            let c2 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p2[1]);
            let c3 = colormap::color_interp(colormap_name, yrange[0], yrange[1], p3[1]);

            colors.push(c0);
            colors.push(c1);
            colors.push(c2);
            colors.push(c2);
            colors.push(c3);
            colors.push(c0);
        }
    }
    (positions, normals, colors)
}
