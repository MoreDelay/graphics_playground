use iced_wgpu::Renderer;
use iced_widget::{bottom, button, column, row, slider, text, text_input};
use iced_winit::core::{Color, Element, Theme};

use crate::scene::SceneWidget;

pub struct Controls {
    background_color: Color,
    input: String,
}

#[derive(Debug, Clone)]
pub enum Message {
    BackgroundColorChanged(Color),
    InputChanged(String),
}

impl Controls {
    pub fn new() -> Self {
        Self {
            background_color: Color::BLACK,
            input: String::default(),
        }
    }

    pub const fn background_color(&self) -> Color {
        self.background_color
    }
}

impl Controls {
    pub fn update(&mut self, message: Message) {
        match message {
            Message::BackgroundColorChanged(color) => {
                self.background_color = color;
            }
            Message::InputChanged(input) => {
                self.input = input;
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        // let bg = self.background_color;
        //
        // let sliders = row![
        //     slider(0.0..=1.0, bg.r, move |r| {
        //         Message::BackgroundColorChanged(Color { r, ..bg })
        //     })
        //     .step(0.01),
        //     slider(0.0..=1.0, bg.g, move |g| {
        //         Message::BackgroundColorChanged(Color { g, ..bg })
        //     })
        //     .step(0.01),
        //     slider(0.0..=1.0, bg.b, move |b| {
        //         Message::BackgroundColorChanged(Color { b, ..bg })
        //     })
        //     .step(0.01),
        // ]
        // .width(500)
        // .spacing(20);
        //
        // bottom(
        //     column![
        //         text("Background color").color(Color::WHITE),
        //         text!("{bg:?}").size(14).color(Color::WHITE),
        //         sliders,
        //         text_input("Type something...", &self.input).on_input(Message::InputChanged),
        //     ]
        //     .spacing(10),
        // )
        // .padding(10)
        // .into()

        row![
            column![text("Hello World").color(Color::WHITE), button("a button")].width(200),
            SceneWidget::new()
        ]
        .into()
    }
}
