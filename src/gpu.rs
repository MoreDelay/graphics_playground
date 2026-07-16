use iced::wgpu;
use wesl::Wesl;

pub const SHADER_ROOT: &str = "src/shader";

pub fn create_simple_shader_module_desc<'a>(
    label: Option<&'a str>,
    wesl_path: &str,
) -> wgpu::ShaderModuleDescriptor<'a> {
    let compute_module = wesl_path.parse().expect("module path invalid");
    let compute_module = Wesl::new(SHADER_ROOT)
        .compile(&compute_module)
        .inspect_err(|e| eprintln!("WESL error: {e}"))
        .expect("shader invalid")
        .to_string();
    wgpu::ShaderModuleDescriptor {
        label,
        source: wgpu::ShaderSource::Wgsl(compute_module.into()),
    }
}
