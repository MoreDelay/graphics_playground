use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use nom::{
    IResult, Offset, Parser,
    branch::alt,
    bytes::streaming::{tag, take_until, take_while1},
    character::streaming::{line_ending, space1, u32 as parse_u32},
    combinator::opt,
    multi::many1,
    number::streaming::float,
    sequence::{delimited, preceded},
};

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r, g, b]
    }
}

use thiserror::Error;

const NS_MIN: f32 = 0.;
const NS_MAX: f32 = 10_000.;

#[derive(Error, Debug)]
pub enum MtlError {
    #[error("could not read file")]
    IoError(#[from] std::io::Error),
    #[error("MTL file {0} is ill-formed at line {1}")]
    WrongFormat(PathBuf, usize),
    #[error("MTL file {0} on line {1}, value is out of spec")]
    OutOfSpec(PathBuf, usize, #[source] MtlSpecError),
}

#[derive(Error, Debug)]
pub enum MtlSpecError {
    #[error("Specified a field twice")]
    DoubleField,
    #[error("Got the same name for a material twice: {0}")]
    Duplicate(String),
    #[error("Setting property before any 'newmtl'")]
    MissingNewmtl,
    #[error("A color value must be between 0 and 1, got {0}")]
    NotAColor(f32),
    #[error("Specular component (Ns) must be within {NS_MIN} and {NS_MAX}, got {0}")]
    NsOutOfRange(f32),
    #[error("Invalid value for illum, only 1 and 2 allowed, got {0}")]
    InvalidIllum(u32),
    #[error("The properties Tr and d must sum to 1, but got Tr={tr} and d={d}")]
    NonOppositeTrAndD { tr: f32, d: f32 },
}

#[derive(Debug, Clone)]
pub struct MtlNewmtl(pub String);

#[derive(Debug, Clone, Copy)]
pub struct MtlKa(pub f32, pub f32, pub f32);

#[derive(Debug, Clone, Copy)]
pub struct MtlKd(pub f32, pub f32, pub f32);

#[derive(Debug, Clone, Copy)]
pub struct MtlKs(pub f32, pub f32, pub f32);

#[derive(Debug, Clone, Copy)]
pub struct MtlD(pub f32);

#[derive(Debug, Clone, Copy)]
pub struct MtlTr(pub f32);

#[derive(Debug, Clone, Copy)]
pub struct MtlNs(pub f32);

#[derive(Debug, Clone, Copy)]
pub enum MtlTransparency {
    None,
    Glass,
    Refraction,
}

#[derive(Debug, Clone, Copy)]
pub enum MtlReflection {
    Simple,
    RayTrace,
    FresnelAndRayTrace,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum MtlIllum {
    Flat,
    Ambient,
    #[default]
    Specular,
    SpecialMode {
        transparency: MtlTransparency,
        reflection: MtlReflection,
    },
    InvisibleSurfaceShadows,
}

#[derive(Debug, Clone)]
pub struct MtlMapKa(pub PathBuf);

#[derive(Debug, Clone)]
pub struct MtlMapBump(pub PathBuf);

#[derive(Debug, Clone)]
pub struct MtlMapKd(pub PathBuf);

#[derive(Debug)]
pub struct ParsedMtl {
    pub name: String,
    pub ka: Option<MtlKa>,
    pub kd: Option<MtlKd>,
    pub ks: Option<MtlKs>,
    pub d: Option<MtlD>,
    pub tr: Option<MtlTr>,
    pub ns: Option<MtlNs>,
    pub illum: Option<MtlIllum>,
    pub map_ka: Option<MtlMapKa>,
    // from here on come inofficial fields that are not defined in the official spec
    pub map_bump: Option<MtlMapBump>,
    pub map_kd: Option<MtlMapKd>,
    // collect other fields just in case
    pub unknown: HashMap<String, String>,
}

impl Default for MtlKa {
    fn default() -> Self {
        Self(0.2, 0.2, 0.2)
    }
}

impl Default for MtlKd {
    fn default() -> Self {
        Self(0.8, 0.8, 0.8)
    }
}

impl Default for MtlKs {
    fn default() -> Self {
        Self(1., 1., 1.)
    }
}

impl Default for MtlD {
    fn default() -> Self {
        Self(1.)
    }
}

impl Default for MtlTr {
    fn default() -> Self {
        Self(0.)
    }
}

impl Default for MtlNs {
    fn default() -> Self {
        Self(0.)
    }
}

impl ParsedMtl {
    fn new(name: String) -> ParsedMtl {
        ParsedMtl {
            name,
            ka: None,
            kd: None,
            ks: None,
            d: None,
            tr: None,
            ns: None,
            illum: None,
            map_ka: None,
            map_bump: None,
            map_kd: None,
            unknown: HashMap::new(),
        }
    }

    fn update(&mut self, line: MtlProperty) -> Result<(), MtlSpecError> {
        use MtlProperty::*;

        match line {
            Ka(ka) if self.ka.is_none() => self.ka = Some(ka),
            Kd(kd) if self.kd.is_none() => self.kd = Some(kd),
            Ks(ks) if self.ks.is_none() => self.ks = Some(ks),
            D(d) if self.d.is_none() => {
                if let Some(MtlTr(tr)) = self.tr
                    && let MtlD(d) = d
                    && (tr + d - 1.).abs() > 1e-05
                {
                    return Err(MtlSpecError::NonOppositeTrAndD { tr, d });
                }
                self.d = Some(d)
            }
            Tr(tr) if self.tr.is_none() => {
                if let Some(MtlD(d)) = self.d
                    && let MtlTr(tr) = tr
                    && (tr + d - 1.).abs() > 1e-05
                {
                    return Err(MtlSpecError::NonOppositeTrAndD { tr, d });
                }
                self.tr = Some(tr)
            }
            Ns(ns) if self.ns.is_none() => self.ns = Some(ns),
            Illum(illum) if self.illum.is_none() => self.illum = Some(illum),
            MapKa(map_ka) if self.map_ka.is_none() => self.map_ka = Some(map_ka),
            MapKd(map_kd) if self.map_kd.is_none() => self.map_kd = Some(map_kd),
            MapBump(map_bump) if self.map_bump.is_none() => self.map_bump = Some(map_bump),
            Unknown(key, value) if !self.unknown.contains_key(&key) => {
                self.unknown.insert(key, value);
            }
            _ => return Err(MtlSpecError::DoubleField),
        };
        Ok(())
    }
}

#[derive(Debug)]
enum MtlProperty {
    Ka(MtlKa),
    Kd(MtlKd),
    Ks(MtlKs),
    D(MtlD),
    Tr(MtlTr),
    Ns(MtlNs),
    Illum(MtlIllum),
    MapKa(MtlMapKa),
    MapBump(MtlMapBump),
    MapKd(MtlMapKd),
    Unknown(String, String),
}

#[derive(Debug)]
enum MtlLine {
    Empty,
    Newmtl(MtlNewmtl),
    Property(Result<MtlProperty, MtlSpecError>),
}

fn mtl_whitespace(input: &str) -> IResult<&str, ()> {
    // allow for line continuation
    let mut parser = many1(alt((space1, preceded(tag("\\"), line_ending))));
    let (input, _whitespace) = parser.parse(input)?;
    Ok((input, ()))
}

fn mtl_comment(input: &str) -> IResult<&str, &str> {
    let mut parser = preceded(tag("#"), take_until("\n"));
    let (input, comment) = parser.parse(input)?;
    Ok((input, comment.trim()))
}

fn mtl_blank_line(input: &str) -> IResult<&str, ()> {
    let mut parser = delimited(opt(mtl_whitespace), opt(mtl_comment), super::end_of_line);
    let (input, _comment) = parser.parse(input)?;
    Ok((input, ()))
}

fn mtl_newmtl(input: &str) -> IResult<&str, MtlNewmtl> {
    let parser = take_while1(|c: char| !c.is_whitespace()).map(|s| MtlNewmtl(String::from(s)));
    preceded((tag("newmtl"), mtl_whitespace), parser).parse(input)
}

fn mtl_color_value(input: &str) -> IResult<&str, Result<f32, MtlSpecError>> {
    let (rest, value) = float.parse(input)?;
    if (0. ..=1.).contains(&value) {
        Ok((rest, Ok(value)))
    } else {
        Ok((input, Err(MtlSpecError::NotAColor(value))))
    }
}

fn mtl_color_triple(input: &str) -> IResult<&str, Result<(f32, f32, f32), MtlSpecError>> {
    (
        mtl_color_value,
        preceded(mtl_whitespace, mtl_color_value),
        preceded(mtl_whitespace, mtl_color_value),
    )
        .map(|(a, b, c)| Ok((a?, b?, c?)))
        .parse(input)
}

fn mtl_ka(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = mtl_color_triple.map(|triple| {
        let (r, g, b) = triple?;
        Ok(MtlProperty::Ka(MtlKa(r, g, b)))
    });
    preceded((tag("Ka"), mtl_whitespace), parser).parse(input)
}

fn mtl_kd(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = mtl_color_triple.map(|triple| {
        let (r, g, b) = triple?;
        Ok(MtlProperty::Kd(MtlKd(r, g, b)))
    });
    preceded((tag("Kd"), mtl_whitespace), parser).parse(input)
}

fn mtl_ks(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = mtl_color_triple.map(|triple| {
        let (r, g, b) = triple?;
        Ok(MtlProperty::Ks(MtlKs(r, g, b)))
    });
    preceded((tag("Ks"), mtl_whitespace), parser).parse(input)
}

fn mtl_d(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = mtl_color_value.map(|v| Ok(MtlProperty::D(MtlD(v?))));
    preceded((tag("d"), mtl_whitespace), parser).parse(input)
}

fn mtl_tr(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = mtl_color_value.map(|v| Ok(MtlProperty::Tr(MtlTr(v?))));
    preceded((tag("Tr"), mtl_whitespace), parser).parse(input)
}

fn mtl_ns(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let parser = float.map(|v| match (NS_MIN..=NS_MAX).contains(&v) {
        true => Ok(MtlProperty::Ns(MtlNs(v))),
        false => Err(MtlSpecError::NsOutOfRange(v)),
    });
    preceded((tag("Ns"), mtl_whitespace), parser).parse(input)
}

fn mtl_illum(input: &str) -> IResult<&str, Result<MtlProperty, MtlSpecError>> {
    let illum = parse_u32.map(|v| match v {
        0 => Ok(MtlProperty::Illum(MtlIllum::Flat)),
        1 => Ok(MtlProperty::Illum(MtlIllum::Ambient)),
        2 => Ok(MtlProperty::Illum(MtlIllum::Specular)),
        3 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::None,
            reflection: MtlReflection::RayTrace,
        })),
        4 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::Glass,
            reflection: MtlReflection::RayTrace,
        })),
        5 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::None,
            reflection: MtlReflection::FresnelAndRayTrace,
        })),
        6 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::Refraction,
            reflection: MtlReflection::RayTrace,
        })),
        7 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::Refraction,
            reflection: MtlReflection::FresnelAndRayTrace,
        })),
        8 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::None,
            reflection: MtlReflection::Simple,
        })),
        9 => Ok(MtlProperty::Illum(MtlIllum::SpecialMode {
            transparency: MtlTransparency::Glass,
            reflection: MtlReflection::Simple,
        })),
        10 => Ok(MtlProperty::Illum(MtlIllum::InvisibleSurfaceShadows)),
        v => Err(MtlSpecError::InvalidIllum(v)),
    });
    preceded((tag("illum"), mtl_whitespace), illum).parse(input)
}

fn mtl_map_ka(input: &str) -> IResult<&str, MtlProperty> {
    let parser = take_while1(|c: char| !c.is_whitespace())
        .map(|s| MtlProperty::MapKa(MtlMapKa(PathBuf::from(s))));
    preceded((tag("map_Ka"), mtl_whitespace), parser).parse(input)
}

fn mtl_map_kd(input: &str) -> IResult<&str, MtlProperty> {
    let parser = take_while1(|c: char| !c.is_whitespace())
        .map(|s| MtlProperty::MapKd(MtlMapKd(PathBuf::from(s))));
    preceded((tag("map_Kd"), mtl_whitespace), parser).parse(input)
}

fn mtl_map_bump(input: &str) -> IResult<&str, MtlProperty> {
    let parser = take_while1(|c: char| !c.is_whitespace())
        .map(|s| MtlProperty::MapBump(MtlMapBump(PathBuf::from(s))));
    preceded((tag("map_Bump"), mtl_whitespace), parser).parse(input)
}

fn mtl_unknown(input: &str) -> IResult<&str, MtlProperty> {
    let mut key = take_while1(|c: char| !c.is_whitespace());
    let mut value = many1(preceded(
        mtl_whitespace,
        take_while1(|c: char| !c.is_whitespace()),
    ));
    let (rest, key) = key.parse(input)?;
    let (rest, value) = value.parse(rest)?;
    Ok((
        rest,
        MtlProperty::Unknown(String::from(key), value.join(" ")),
    ))
}

pub fn parse_mtl(path: &Path) -> Result<Vec<ParsedMtl>, MtlError> {
    let file = File::open(path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut all_materials: Vec<ParsedMtl> = Vec::new();
    let mut current_mtl = None;

    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let parser = alt((
                mtl_blank_line.map(|_| MtlLine::Empty),
                mtl_newmtl.map(MtlLine::Newmtl),
                mtl_ka.map(MtlLine::Property),
                mtl_kd.map(MtlLine::Property),
                mtl_ks.map(MtlLine::Property),
                mtl_d.map(MtlLine::Property),
                mtl_tr.map(MtlLine::Property),
                mtl_ns.map(MtlLine::Property),
                mtl_illum.map(MtlLine::Property),
                mtl_map_ka.map(|v| MtlLine::Property(Ok(v))),
                mtl_map_bump.map(|v| MtlLine::Property(Ok(v))),
                mtl_map_kd.map(|v| MtlLine::Property(Ok(v))),
                mtl_unknown.map(|v| MtlLine::Property(Ok(v))),
            ));
            let mut parser = preceded(opt(mtl_whitespace), parser);
            let (rest, line) = match parser.parse(&buffer) {
                Ok((rest, line)) => (rest, line),
                Err(nom::Err::Incomplete(_)) => break,
                Err(_) => return Err(MtlError::WrongFormat(path.to_path_buf(), line_index)),
            };
            drop(parser);
            let offset = buffer.offset(rest);
            buffer = buffer.split_off(offset);

            let map_spec_err = |e| MtlError::OutOfSpec(path.to_path_buf(), line_index, e);

            match line {
                MtlLine::Empty => (),
                MtlLine::Newmtl(MtlNewmtl(name)) => {
                    if let Some(last_mtl) = current_mtl {
                        all_materials.push(last_mtl);
                    }
                    let duplicate = all_materials.iter().any(|mtl| mtl.name == name);
                    if duplicate {
                        let e = MtlSpecError::Duplicate(name);
                        return Err(map_spec_err(e));
                    }
                    current_mtl = Some(ParsedMtl::new(name))
                }
                MtlLine::Property(prop) => {
                    let prop = prop.map_err(map_spec_err)?;
                    let Some(current_mtl) = current_mtl.as_mut() else {
                        let e = map_spec_err(MtlSpecError::MissingNewmtl);
                        return Err(e);
                    };
                    current_mtl.update(prop).map_err(map_spec_err)?;
                }
            }
        }
    }

    if let Some(mtl) = current_mtl {
        all_materials.push(mtl);
    }

    Ok(all_materials)
}

impl From<MtlKa> for Color {
    fn from(MtlKa(r, g, b): MtlKa) -> Self {
        Color { r, g, b }
    }
}

impl From<MtlKd> for Color {
    fn from(MtlKd(r, g, b): MtlKd) -> Self {
        Color { r, g, b }
    }
}

impl From<MtlKs> for Color {
    fn from(MtlKs(r, g, b): MtlKs) -> Self {
        Color { r, g, b }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_parse_mtl() {
        let path = PathBuf::from("./resources/cube/cube.mtl");
        let mtl = parse_mtl(&path).unwrap();
        let mtl = mtl.into_iter().next().unwrap();
        assert!(mtl.name == "Material.001");
        assert!(matches!(mtl.ns, Some(MtlNs(323.999994))));
        assert!(matches!(mtl.kd, Some(MtlKd(0.8, 0.8, 0.8))));
        let MtlMapBump(file) = mtl.map_bump.unwrap();
        assert!(file.file_name().unwrap() == "cube-normal.png");
    }
}
