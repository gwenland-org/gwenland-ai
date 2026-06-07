#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Plugins used by the Settings screen:
        //   fs     — read/write ~/.config/gwen/config.json and clear session files
        //   shell  — open the config/models/sessions folders in the OS file
        //            explorer and run `gwen update --check`
        //   dialog — folder picker (reserved for future "browse" path rows)
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error running GwenLand");
}
