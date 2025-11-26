use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use nom::branch::alt;
use nom::character::streaming::line_ending;
use nom::combinator::eof;
use nom::{IResult, Parser};

pub mod mtl;
pub mod obj;

pub use mtl::ParsedMtl;
pub use obj::ParsedObj;

pub trait MtlManager {
    /// Request to load the provided MTL library. The argument is just the provided name within the
    /// file which typically is concatenated with the parent directory of the original obj file.
    fn request_mtl_load(&mut self, name: &str) -> Result<HashSet<String>, mtl::MtlError>;

    /// Given the name of a material, this call should return the index to the material that
    /// uniquely identifies the corresponding material for this manager.
    fn request_mtl_index(&self, name: &str) -> Option<u32>;
}

pub struct SimpleMtlManager {
    root_dir: PathBuf,
    materials: Vec<mtl::ParsedMtl>,
    name_mapping: HashMap<String, u32>,
}

impl SimpleMtlManager {
    pub fn new(root_dir: PathBuf) -> Self {
        Self {
            root_dir,
            materials: Vec::new(),
            name_mapping: HashMap::new(),
        }
    }

    pub fn extract_list(self) -> Vec<mtl::ParsedMtl> {
        self.materials
    }
}

impl MtlManager for SimpleMtlManager {
    fn request_mtl_load(&mut self, name: &str) -> Result<HashSet<String>, mtl::MtlError> {
        let path = self.root_dir.join(name);
        let loaded_mtls = mtl::parse_mtl(&path)?;
        let mut mtls_in_this_lib = HashSet::new();

        let new_root = path.parent().expect("file lies within a folder");

        for mtl in loaded_mtls {
            mtls_in_this_lib.insert(mtl.name.clone());

            use std::collections::hash_map::Entry;
            match self.name_mapping.entry(mtl.name.clone()) {
                Entry::Occupied(_) => continue,
                Entry::Vacant(entry) => {
                    let mut mtl = mtl;
                    if let Some(map_ka) = &mut mtl.map_ka {
                        map_ka.0 = new_root.join(&map_ka.0);
                    }
                    if let Some(map_kd) = &mut mtl.map_kd {
                        map_kd.0 = new_root.join(&map_kd.0);
                    }
                    if let Some(map_bump) = &mut mtl.map_bump {
                        map_bump.0 = new_root.join(&map_bump.0);
                    }
                    let index = self.materials.len() as u32;
                    entry.insert(index);
                    self.materials.push(mtl);
                }
            }
        }

        Ok(mtls_in_this_lib)
    }

    fn request_mtl_index(&self, name: &str) -> Option<u32> {
        self.name_mapping.get(name).copied()
    }
}

fn end_of_line(input: &str) -> IResult<&str, ()> {
    let mut parser = alt((line_ending, eof));
    let (input, _) = parser.parse(input)?;
    Ok((input, ()))
}
