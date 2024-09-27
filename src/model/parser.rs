use nom::branch::alt;
use nom::bytes::complete::{tag, take_until};
use nom::character::complete::{line_ending, space0, space1};
use nom::combinator::{eof, opt, verify};
use nom::multi::{fill, fold_many_m_n};
use nom::number::complete::float;
use nom::sequence::{delimited, preceded};
use nom::IResult;

fn end_of_line(input: &str) -> IResult<&str, ()> {
    let mut parser = alt((line_ending, eof));
    let (input, _) = parser(input)?;
    Ok((input, ()))
}

pub fn obj_comment(input: &str) -> IResult<&str, &str> {
    let mut parser = preceded(tag("#"), take_until("\n"));
    let (input, comment) = parser(input)?;
    Ok((input, comment.trim()))
}

pub fn obj_goto_next_line(input: &str) -> IResult<&str, ()> {
    let mut parser = delimited(space0, opt(obj_comment), end_of_line);
    let (input, _comment) = parser(input)?;
    Ok((input, ()))
}

pub fn obj_geometric_vertex(input: &str) -> IResult<&str, Vec<f32>> {
    let parse_numbers = fold_many_m_n(
        3,
        4,
        preceded(space1, float),
        || Vec::with_capacity(4),
        |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        },
    );
    let mut parser = delimited(tag("v"), parse_numbers, obj_goto_next_line);
    let (input, mut vertex) = parser(input)?;

    vertex.resize(4, 1.0); // optional 4th element is 1.0 by default
    Ok((input, vertex))
}

pub fn obj_texture_coordinates(input: &str) -> IResult<&str, Vec<f32>> {
    let correct_tex_coord = verify(float, |&num| 0. <= num && num <= 1.);
    let parse_numbers = fold_many_m_n(
        3,
        4,
        preceded(space1, correct_tex_coord),
        || Vec::with_capacity(4),
        |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        },
    );
    let mut parser = delimited(tag("vt"), parse_numbers, obj_goto_next_line);
    let (input, mut vertex) = parser(input)?;

    vertex.resize(4, 0.0); // optional 4th element is 0.0 by default
    Ok((input, vertex))
}

pub fn obj_vertex_normal(input: &str) -> IResult<&str, Vec<f32>> {
    let mut normal = vec![0.; 3];
    // use closure to satisfy Fn requirement, gets changed to FnMut in nom v8.0.0
    let parse_numbers = fill(|s| preceded(space1, float)(s), &mut normal);
    let mut parser = delimited(tag("vn"), parse_numbers, obj_goto_next_line);
    let (input, ()) = parser(input)?;

    drop(parser); // release mut ref to normal
    Ok((input, normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obj_geometric_vertex() {
        let input = "v 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_geometric_vertex(input).unwrap();
        assert_eq!(rest, "");
        assert_eq!(vec, vec![1.1, 2.2, 3.3, 1.0]);
    }

    #[test]
    fn test_obj_texture_coordinates() {
        let input = "vt 1.0   0.2   0.3 # this is a comment\n";
        let (rest, vec) = obj_texture_coordinates(input).unwrap();
        assert_eq!(rest, "");
        assert_eq!(vec, vec![1.0, 0.2, 0.3, 0.0]);
    }

    #[test]
    fn test_obj_texture_coordinates_fail() {
        let input = "vt 1.1   2.2   3.3 # this is a comment\n";
        let out = obj_texture_coordinates(input);
        assert!(out.is_err());
    }

    #[test]
    fn test_obj_vertex_normal() {
        let input = "vn 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_vertex_normal(input).unwrap();
        assert_eq!(rest, "");
        assert_eq!(vec, vec![1.1, 2.2, 3.3]);
    }
}
