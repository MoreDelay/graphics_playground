use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader},
    ops::Range,
    path::{Path, PathBuf},
};

use nom::{
    IResult, Offset, Parser,
    branch::alt,
    bytes::streaming::{tag, take_until, take_while1},
    character::streaming::{i32 as parse_i32, line_ending, space1},
    combinator::opt,
    multi::{fill, many_m_n, many1, many1_count},
    number::streaming::float,
    sequence::{delimited, preceded},
};

use thiserror::Error;

use super::end_of_line;

const MAX_VERTICES_PER_FACE: usize = 16;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct V {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vt {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vn {
    pub i: f32,
    pub j: f32,
    pub k: f32,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct FTriplet {
    pub index_vertex: u32,
    pub index_texture: Option<u32>,
    pub index_normal: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct F {
    pub triplets: Vec<FTriplet>,
}

#[derive(Debug, Clone)]
pub struct FaceRange(pub Range<u32>);

#[derive(Clone)]
pub struct Object {
    pub name: Option<String>,
    pub faces: Vec<F>,
    pub groups: Vec<(String, FaceRange)>,
    pub mtls: Vec<(Option<u32>, FaceRange)>,
}

#[derive(Clone)]
pub struct ParsedObj {
    pub geo_vertices: Vec<V>,
    pub tex_vertices: Vec<Vt>,
    pub vertex_normals: Vec<Vn>,
    pub objects: Vec<Object>,
}

#[derive(Error, Debug)]
pub enum ObjError {
    #[error("could not read file")]
    IoError(#[from] std::io::Error),
    #[error("OBJ file {0} is ill-formed at line {1}")]
    WrongFormat(PathBuf, usize),
    #[error("Obj file {0} on line {1}, value is out of spec")]
    OutOfSpec(PathBuf, usize, #[source] ObjSpecError),
    #[error("issue with MTL file")]
    FromMtl(#[source] super::mtl::MtlError),
}

#[derive(Error, Debug)]
pub enum ObjSpecError {
    #[error("invalid list index, must be non-zero and expect between -{0} and {0}, got {1}")]
    InvalidListIndex(u32, i32),
    #[error(
        "a face must have at least 3 vertices, supported at most {MAX_VERTICES_PER_FACE}, found {0}"
    )]
    BadFace(usize),
}

#[derive(Debug, Clone)]
struct G {
    names: Vec<String>,
}

#[derive(Debug, Clone)]
struct O {
    name: String,
}

#[derive(Debug, PartialEq, Clone)]
struct Mtllib {
    pub name: String,
}

#[derive(Debug, PartialEq, Clone)]
struct Usemtl {
    name: Option<String>,
}

enum Line {
    Empty,
    V(V),
    Vt(Result<Vt, ObjSpecError>),
    Vn(Vn),
    F(Result<F, ObjSpecError>),
    G(G),
    O(O),
    MtlLib(Mtllib),
    UseMtl(Usemtl),
}

fn obj_whitespace(input: &str) -> IResult<&str, ()> {
    // allow for line continuation
    let mut parser = many1_count(alt((space1, preceded(tag("\\"), line_ending))));
    let (rest, _ws_count) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_comment(input: &str) -> IResult<&str, &str> {
    let mut parser = preceded(tag("#"), take_until("\n"));
    let (rest, comment) = parser.parse(input)?;
    Ok((rest, comment.trim()))
}

fn obj_blank_line(input: &str) -> IResult<&str, ()> {
    let mut parser = delimited(opt(obj_whitespace), opt(obj_comment), end_of_line);
    let (rest, _comment) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_ignore(input: &str) -> IResult<&str, ()> {
    let mut parser = preceded(alt((tag("o"), tag("s"))), take_until("\n"));
    let (rest, _) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_v(input: &str) -> IResult<&str, V> {
    let parser = (
        preceded(obj_whitespace, float),
        preceded(obj_whitespace, float),
        preceded(obj_whitespace, float),
        opt(preceded(obj_whitespace, float)),
    );
    let mut parser = delimited(tag("v"), parser, obj_blank_line);
    let (rest, (x, y, z, w)) = parser.parse(input)?;

    let w = w.unwrap_or(1.);
    let vertex = V { x, y, z, w };
    Ok((rest, vertex))
}

fn obj_vt(input: &str) -> IResult<&str, Result<Vt, ObjSpecError>> {
    let parser = (
        preceded(obj_whitespace, float),
        opt(preceded(obj_whitespace, float)),
        opt(preceded(obj_whitespace, float)),
    );
    let mut parser = delimited(tag("vt"), parser, obj_blank_line);
    let (rest, (u, v, w)) = parser.parse(input)?;

    let v = v.unwrap_or(0.);
    let w = w.unwrap_or(0.);

    let vertex = Vt { u, v, w };

    Ok((rest, Ok(vertex)))
}

fn obj_vn(input: &str) -> IResult<&str, Vn> {
    let mut normal = [0.; 3];
    let parse_numbers = fill(preceded(obj_whitespace, float), &mut normal);
    let mut parser = delimited(tag("vn"), parse_numbers, obj_blank_line);
    let (rest, ()) = parser.parse(input)?;
    drop(parser); // release mut ref to normal

    let [i, j, k] = normal;
    let normal = Vn { i, j, k };
    Ok((rest, normal))
}

fn obj_list_index(len: u32) -> impl FnMut(&str) -> IResult<&str, Result<u32, ObjSpecError>> {
    move |i: &str| {
        let (rest, index) = parse_i32.parse(i)?;
        // obj uses 1-based indexing, convert here to 0-based
        let valid = index != 0 && index.unsigned_abs() <= len;
        if !valid {
            return Ok((i, Err(ObjSpecError::InvalidListIndex(len, index))));
        }
        let abs = index.unsigned_abs();
        let index = if index < 0 { len - abs } else { abs - 1 };
        Ok((rest, Ok(index)))
    }
}

fn obj_face_vxx(
    n_vertices: u32,
) -> impl FnMut(&str) -> IResult<&str, Result<FTriplet, ObjSpecError>> {
    move |i: &str| {
        obj_list_index(n_vertices)
            .map(|v| {
                Ok(FTriplet {
                    index_vertex: v?,
                    index_texture: None,
                    index_normal: None,
                })
            })
            .parse(i)
    }
}

fn obj_face_vtx(
    n_vertices: u32,
    n_textures: u32,
) -> impl FnMut(&str) -> IResult<&str, Result<FTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("/"),
            obj_list_index(n_textures),
        )
            .map(|(v, _, t)| {
                Ok(FTriplet {
                    index_vertex: v?,
                    index_texture: Some(t?),
                    index_normal: None,
                })
            })
            .parse(i)
    }
}

fn obj_face_vxn(
    n_vertices: u32,
    n_normals: u32,
) -> impl FnMut(&str) -> IResult<&str, Result<FTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("//"),
            obj_list_index(n_normals),
        )
            .map(|(v, _, n)| {
                Ok(FTriplet {
                    index_vertex: v?,
                    index_texture: None,
                    index_normal: Some(n?),
                })
            })
            .parse(i)
    }
}

fn obj_face_vtn(
    n_vertices: u32,
    n_textures: u32,
    n_normals: u32,
) -> impl FnMut(&str) -> IResult<&str, Result<FTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("/"),
            obj_list_index(n_textures),
            tag("/"),
            obj_list_index(n_normals),
        )
            .map(|(v, _, t, _, n)| {
                Ok(FTriplet {
                    index_vertex: v?,
                    index_texture: Some(t?),
                    index_normal: Some(n?),
                })
            })
            .parse(i)
    }
}

fn obj_f(n_v: u32, n_t: u32, n_n: u32) -> impl Fn(&str) -> IResult<&str, Result<F, ObjSpecError>>
where
{
    const MAX: usize = MAX_VERTICES_PER_FACE;

    move |input: &str| {
        let counter = alt((
            preceded(
                (obj_whitespace, obj_face_vxx(n_v)),
                many1_count(preceded(obj_whitespace, obj_face_vxx(n_v))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vtx(n_v, n_t)),
                many1_count(preceded(obj_whitespace, obj_face_vtx(n_v, n_t))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vxn(n_v, n_n)),
                many1_count(preceded(obj_whitespace, obj_face_vxn(n_v, n_n))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vtn(n_v, n_t, n_n)),
                many1_count(preceded(obj_whitespace, obj_face_vtn(n_v, n_t, n_n))).map(|v| v + 1),
            ),
        ));
        let mut counter = delimited(tag("f"), counter, obj_blank_line);
        let (_, count) = counter.parse(input)?;
        let valid = (3..=MAX).contains(&count);
        if !valid {
            return Ok((input, Err(ObjSpecError::BadFace(count))));
        }
        let parser = alt((
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vxx(n_v))),
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vtx(n_v, n_t))),
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vxn(n_v, n_n))),
            many_m_n(
                3,
                MAX,
                preceded(obj_whitespace, obj_face_vtn(n_v, n_t, n_n)),
            ),
        ));
        let mut parser = delimited(tag("f"), parser, obj_blank_line);
        let (rest, triplets) = parser.parse(input)?;

        let face = match triplets.into_iter().collect::<Result<_, _>>() {
            Ok(triplets) => F { triplets },
            Err(e) => return Ok((input, Err(e))),
        };
        Ok((rest, Ok(face)))
    }
}

fn obj_mtllib(input: &str) -> IResult<&str, Mtllib> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("mtllib"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.to_string();
    let mtllib = Mtllib { name };
    Ok((input, mtllib))
}

fn obj_g(input: &str) -> IResult<&str, G> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let parser = many1(parser);
    let mut parser = delimited(tag("g"), parser, obj_blank_line);

    let (input, names) = parser.parse(input)?;
    let names = names.into_iter().map(String::from).collect();
    let group = G { names };
    Ok((input, group))
}

fn obj_usemtl(input: &str) -> IResult<&str, Usemtl> {
    let parser = opt(take_while1(|c: char| !c.is_whitespace()));
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("usemtl"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.map(|s| s.to_string());
    let usemtl = Usemtl { name };
    Ok((input, usemtl))
}

fn obj_o(input: &str) -> IResult<&str, O> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("o"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.to_string();
    let object = O { name };
    Ok((input, object))
}

fn obj_line(
    n_vertices: u32,
    n_textures: u32,
    n_normals: u32,
) -> impl FnMut(&str) -> IResult<&str, Line> {
    move |i: &str| {
        alt((
            obj_blank_line.map(|_| Line::Empty),
            obj_v.map(Line::V),
            obj_vt.map(Line::Vt),
            obj_vn.map(Line::Vn),
            obj_f(n_vertices, n_textures, n_normals).map(Line::F),
            obj_mtllib.map(Line::MtlLib),
            obj_g.map(Line::G),
            obj_o.map(Line::O),
            obj_usemtl.map(Line::UseMtl),
            obj_ignore.map(|_| Line::Empty),
        ))
        .parse(i)
    }
}

impl ParsedObj {
    // Only used internally
    fn new_empty() -> Self {
        Self {
            geo_vertices: Vec::new(),
            tex_vertices: Vec::new(),
            vertex_normals: Vec::new(),
            objects: Vec::new(),
        }
    }
}

pub fn load_obj(
    path: &Path,
    mtl_manager: &mut impl super::MtlManager,
) -> Result<ParsedObj, ObjError> {
    let file = File::open(path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut res = ParsedObj::new_empty();

    // parser state helper struct
    struct GroupState {
        names: Vec<String>,
        start: u32,
    }
    struct MtlState {
        index: Option<u32>,
        start: u32,
    }

    // actual parser state
    let mut cur_object = Object {
        name: None,
        faces: Vec::new(),
        groups: Vec::new(),
        mtls: Vec::new(),
    };
    let mut cur_groups: Option<GroupState> = None;
    let mut cur_mtl = MtlState {
        index: None,
        start: 0,
    };

    let mut material_names = HashSet::new();

    // parser loop
    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let n_v = res.geo_vertices.len() as u32;
            let n_t = res.tex_vertices.len() as u32;
            let n_n = res.vertex_normals.len() as u32;
            let (rest, line) = match obj_line(n_v, n_t, n_n).parse(&buffer) {
                Ok((rest, line)) => (rest, line),
                Err(nom::Err::Incomplete(_)) => break,
                _ => return Err(ObjError::WrongFormat(path.to_path_buf(), line_index)),
            };
            let offset = buffer.offset(rest);
            buffer = buffer.split_off(offset);

            let map_spec_err = |e| ObjError::OutOfSpec(path.to_path_buf(), line_index, e);

            match line {
                Line::Empty => (),
                Line::V(obj_v) => res.geo_vertices.push(obj_v),
                Line::Vt(obj_vt) => res.tex_vertices.push(obj_vt.map_err(map_spec_err)?),
                Line::Vn(obj_vn) => res.vertex_normals.push(obj_vn),
                Line::F(obj_f) => cur_object.faces.push(obj_f.map_err(map_spec_err)?),
                Line::MtlLib(Mtllib { name }) => {
                    let found_names = mtl_manager
                        .request_mtl_load(&name)
                        .map_err(ObjError::FromMtl)?;
                    material_names.extend(found_names);
                }
                Line::G(G { names }) => {
                    // complete current groups
                    let end = cur_object.faces.len() as u32;
                    if let Some(cur_groups) = cur_groups.take()
                        && cur_groups.start < end
                    {
                        let range = FaceRange(cur_groups.start..end);
                        for name in cur_groups.names {
                            cur_object.groups.push((name, range.clone()));
                        }
                    }

                    // start new groups
                    let start = end;
                    cur_groups = Some(GroupState { names, start });
                }
                Line::UseMtl(Usemtl { name }) => {
                    // complete current material
                    let end = cur_object.faces.len() as u32;
                    if cur_mtl.start < end {
                        let range = FaceRange(cur_mtl.start..end);
                        cur_object.mtls.push((cur_mtl.index, range));
                    }

                    // start new material
                    let start = end;
                    let index = name
                        .map(|name| {
                            let valid_name = material_names.contains(&name);
                            if !valid_name {
                                return Err(ObjError::WrongFormat(path.to_path_buf(), line_index));
                            }
                            Ok(mtl_manager
                                .request_mtl_index(&name)
                                .expect("we must have stored this material in the manager before"))
                        })
                        .transpose()?;
                    cur_mtl = MtlState { index, start };
                }
                Line::O(O { name: new_name }) => {
                    // complete current object
                    res.objects.push(cur_object);

                    // start new object
                    cur_object = Object {
                        name: Some(new_name),
                        faces: Vec::new(),
                        groups: Vec::new(),
                        mtls: Vec::new(),
                    };
                }
            };
        }
    }

    // complete final object
    let end = cur_object.faces.len() as u32;

    if let Some(cur_groups) = cur_groups.take() {
        let range = FaceRange(cur_groups.start..end);
        for name in cur_groups.names {
            cur_object.groups.push((name, range.clone()));
        }
    }

    let range = FaceRange(cur_mtl.start..end);
    cur_object.mtls.push((cur_mtl.index, range));

    if !cur_object.faces.is_empty() {
        res.objects.push(cur_object);
    }

    Ok(res)
}

#[cfg(test)]
mod tests {
    use crate::SimpleMtlManager;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_obj_blank_line() {
        let input = "# blank line\n# second blank line\n";
        let (rest, _) = obj_blank_line(input).unwrap();
        assert_eq!(rest, "# second blank line\n");
        let (rest, _) = obj_blank_line(rest).unwrap();
        assert_eq!(rest, "");
    }

    #[test]
    fn test_obj_geometric_vertex() {
        let input = "v 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_v(input).unwrap();
        let expected = V {
            x: 1.1,
            y: 2.2,
            z: 3.3,
            w: 1.0,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_texture_coordinates() {
        let input = "vt 1.0   0.2   0.3 # this is a comment\n";
        let (rest, vec) = obj_vt(input).unwrap();
        let vec = vec.unwrap();
        let expected = Vt {
            u: 1.0,
            v: 0.2,
            w: 0.3,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_vertex_normal() {
        let input = "vn 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_vn(input).unwrap();
        let expected = Vn {
            i: 1.1,
            j: 2.2,
            k: 3.3,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_face_element() {
        let input = "f -4/1/1   2/-3/2   3/3/-2 # this is a comment\n";
        let (rest, vec) = obj_f(4, 4, 4).parse(input).unwrap();
        let vec = vec.unwrap();
        let expected = F {
            triplets: vec![
                FTriplet {
                    index_vertex: 0,
                    index_texture: Some(0),
                    index_normal: Some(0),
                },
                FTriplet {
                    index_vertex: 1,
                    index_texture: Some(1),
                    index_normal: Some(1),
                },
                FTriplet {
                    index_vertex: 2,
                    index_texture: Some(2),
                    index_normal: Some(2),
                },
            ],
        };
        assert_eq!(rest, "");
        vec.triplets
            .iter()
            .zip(expected.triplets.iter())
            .for_each(|(got, expected)| assert_eq!(got, expected));
    }

    #[test]
    fn test_obj_face_element_fail() {
        let input = "f 1//1   2//2   3/3/3 # this is a comment\n";
        let out = obj_f(4, 4, 4)(input);
        assert!(out.is_err());
    }

    #[ignore]
    #[test]
    fn test_parse_obj() {
        let path = PathBuf::from("./resources/cube/cube.obj");
        let mut mtl_manager = SimpleMtlManager::new(PathBuf::from("./resources/cube"));
        let parsed = load_obj(&path, &mut mtl_manager).unwrap();
        assert!(parsed.geo_vertices.len() == 216);
        assert!(parsed.tex_vertices.len() == 277);
        assert!(parsed.vertex_normals.len() == 216);
        assert!(parsed.objects.len() == 1);
        let object = &parsed.objects[0];
        assert!(object.faces.len() == 218);
    }
}
