#![allow(dead_code)]

mod parser;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MeshError {
    #[error("could not read file")]
    IoError(#[from] std::io::Error),
    #[error("file is ill-formed")]
    WrongFormat,
}

pub struct VertexIndex(usize);
pub struct FaceIndex(usize);
pub struct HalfEdgeIndex(usize);

pub struct HalfEdge {
    vend: VertexIndex,
    face: FaceIndex,
    next: HalfEdgeIndex,
    opposite: HalfEdgeIndex,
}

pub struct Vertex {
    x: i32,
    y: i32,
    z: i32,
    halfedge: HalfEdgeIndex,
}

pub struct Face {
    halfedge: HalfEdgeIndex,
}

pub struct Mesh {
    halfedges: Vec<HalfEdge>,
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
}

impl Mesh {
    pub fn load_obj(file: PathBuf) -> Result<Self, MeshError> {
        let file = File::open(file)?;
        let mut buffer = String::new();
        let mut reader = BufReader::new(file);
        loop {
            reader.read_line(&mut buffer)?;

            todo!();
        }
    }
}
