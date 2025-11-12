use anyhow::Result;
use eframe::{
    egui::{self, CentralPanel, ScrollArea, TextEdit, TopBottomPanel},
    App, Frame,
};
use rumma_core::AwqModel;
use std::{
    sync::{mpsc, Arc},
    thread,
};

pub enum GuiMessage {
    Status(String),
    Loaded(Result<Arc<AwqModel>>),
}

pub struct RummaApp {
    pub repo_url: String,
    pub status_text: String,
    pub rx: mpsc::Receiver<GuiMessage>,
    pub tx: mpsc::Sender<GuiMessage>,
    pub is_loading: bool,
    pub model: Option<Arc<AwqModel>>,
}

impl Default for RummaApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            repo_url: "Qwen/Qwen1.5-0.5B-Chat-AWQ".to_string(),
            status_text: "Enter a Hugging Face repo URL and click 'Load Model'.".to_string(),
            tx,
            rx,
            is_loading: false,
            model: None,
        }
    }
}

impl App for RummaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        // Check for messages from the loading thread
        if let Ok(msg) = self.rx.try_recv() {
            match msg {
                GuiMessage::Status(status) => {
                    self.status_text.push_str(&format!("\n{}", status));
                }
                GuiMessage::Loaded(Ok(model)) => {
                    self.status_text.push_str("\nModel loaded successfully!");
                    self.model = Some(model);
                    self.is_loading = false;
                }
                GuiMessage::Loaded(Err(e)) => {
                    self.status_text
                        .push_str(&format!("\nError loading model: {}", e));
                    self.is_loading = false;
                }
            }
        }

        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.heading("Rumma AWQ Model Runner");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Hugging Face Repo URL:");
                let url_input =
                    ui.add_enabled(!self.is_loading, TextEdit::singleline(&mut self.repo_url));
                if url_input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    // Handle Enter key press if needed
                }

                if ui
                    .add_enabled(!self.is_loading, egui::Button::new("Load Model"))
                    .clicked()
                {
                    self.is_loading = true;
                    self.status_text = format!("Loading model from: {}...", self.repo_url);
                    let tx = self.tx.clone();
                    let repo_url = self.repo_url.clone();

                    thread::spawn(move || {
                        let result =
                            crate::resolve_and_load_model(&repo_url, Some(tx.clone())).map_err(
                                |e| {
                                    tx.send(GuiMessage::Status(format!("Error: {}", e)))
                                        .unwrap();
                                    e
                                },
                            );
                        tx.send(GuiMessage::Loaded(result.map(Arc::new))).unwrap();
                    });
                }
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            ui.label("Status:");
            ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::multiline(&mut self.status_text)
                            .interactive(false)
                            .font(egui::TextStyle::Monospace),
                    );
                });
        });
    }
}

pub fn run() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Rumma GUI",
        options,
        Box::new(|_cc| Ok(Box::new(RummaApp::default()))),
    )
}
