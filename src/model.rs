#![allow(dead_code)]

mod parser;

use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MeshError {
    #[error("parsing error")]
    Parsing(#[from] parser::ParseError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfEdgeIndex(usize);

impl std::ops::Index<HalfEdgeIndex> for Vec<HalfEdge> {
    type Output = HalfEdge;
    fn index(&self, index: HalfEdgeIndex) -> &Self::Output {
        &self[index.0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceIndex(usize);

impl std::ops::Index<FaceIndex> for Vec<Face> {
    type Output = Face;
    fn index(&self, index: FaceIndex) -> &Self::Output {
        &self[index.0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexIndex(usize);

impl std::ops::Index<VertexIndex> for Vec<Vertex> {
    type Output = Vertex;
    fn index(&self, index: VertexIndex) -> &Self::Output {
        &self[index.0]
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct HalfEdge {
    vend: VertexIndex,
    face: FaceIndex,
    next: HalfEdgeIndex,
    opposite: Option<HalfEdgeIndex>,
}

#[derive(Clone, Copy)]
pub struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    halfedge: Option<HalfEdgeIndex>,
}

#[derive(Clone, Copy)]
pub struct Face {
    halfedge: HalfEdgeIndex,
}

pub struct HalfEdgeMesh {
    halfedges: Vec<HalfEdge>,
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
}

impl HalfEdgeMesh {
    pub fn load_obj(file: &PathBuf) -> Result<Self, MeshError> {
        let parsed_obj = parser::parse_obj(file)?;

        let mut vertices: Vec<_> = parsed_obj
            .vertices
            .iter()
            .map(|v| Vertex {
                x: v.x,
                y: v.y,
                z: v.z,
                halfedge: None,
            })
            .collect();

        let mut halfedges = Vec::new();
        let mut faces = Vec::new();

        for face in parsed_obj.faces {
            assert!(face.triplets.len() > 0);
            let new_face_index = FaceIndex(faces.len());
            let first_new_he_index = HalfEdgeIndex(halfedges.len());

            faces.push(Face {
                halfedge: HalfEdgeIndex(halfedges.len()),
            });
            for triplet in face.triplets {
                let current_he_index = HalfEdgeIndex(halfedges.len());
                let next_he_index = HalfEdgeIndex(current_he_index.0 + 1);
                halfedges.push(HalfEdge {
                    vend: VertexIndex(triplet.vertex),
                    face: new_face_index,
                    next: next_he_index,
                    opposite: None,
                });

                if let None = vertices[triplet.vertex].halfedge {
                    vertices[triplet.vertex].halfedge = Some(current_he_index);
                }
            }

            let last_new_he = halfedges.last_mut().unwrap();
            last_new_he.next = first_new_he_index;
        }

        // create helper data structure to search for needed half edges more easily
        let helper: Vec<_> = faces
            .iter()
            .flat_map(|f| {
                let mut tuples = Vec::new();
                let start = halfedges[f.halfedge];
                let mut he = halfedges[start.next];
                tuples.push((start.vend, he.vend, start.next));
                while he != start {
                    let prev = he;
                    he = halfedges[he.next];
                    let from = prev.vend;
                    let to = he.vend;
                    let he_index = prev.next;
                    tuples.push((from, to, he_index));
                }
                tuples
            })
            .collect();

        // set all opposite half edges
        for he_index in 0..halfedges.len() {
            let start_index = HalfEdgeIndex(he_index);
            let mut iter_he_index = halfedges[he_index].next;
            while halfedges[iter_he_index].next != start_index {
                iter_he_index = halfedges[iter_he_index].next;
            }

            let from = halfedges[iter_he_index].vend;
            let to = halfedges[start_index].vend;
            if let Some(opposite) = helper
                .iter()
                .find_map(|&(f, t, h)| (f == to && t == from).then(|| h))
            {
                halfedges[start_index.0].opposite = Some(opposite);
            }
        }

        Ok(HalfEdgeMesh {
            halfedges,
            vertices,
            faces,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_obj() {
        let file = PathBuf::from("./resources/cube.obj");
        let ds = HalfEdgeMesh::load_obj(&file).unwrap();

        assert_eq!(ds.vertices.len(), 8);
        assert_eq!(ds.faces.len(), 6);
        assert_eq!(ds.halfedges.len(), 24);

        let corner = ds
            .vertices
            .iter()
            .find(|&v| v.x == 1. && v.y == 1. && v.z == 1.);
        assert!(corner.is_some());
        let corner = corner.unwrap();
        let mut he_index = corner.halfedge.unwrap();
        let start = he_index;

        // cube consists of quads
        for _ in 0..3 {
            he_index = ds.halfedges[he_index].next;
            assert_ne!(start, he_index);
        }
        he_index = ds.halfedges[he_index].next;
        assert_eq!(start, he_index);

        let opp = ds.halfedges[he_index].opposite.unwrap();
        let opp_opp = ds.halfedges[opp].opposite.unwrap();
        assert_ne!(he_index, opp);
        assert_eq!(he_index, opp_opp);
    }
}
