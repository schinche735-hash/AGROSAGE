# AgriSage — Smart Crop Disease Detection

A local web app for crop-leaf disease screening with English, Hindi and Marathi interface, disease reference images, cached guidance, and a trained MobileNetV2 model.

## Start it with one command

**Requirements (install once):** Git, Python 3.11 or 3.12, and VS Code. The first run needs an internet connection to install Python packages.

Open a terminal and run this single command:

```bash
git clone https://github.com/schinche735-hash/AGROSAGE.git && cd AGROSAGE && bash setup.sh
```

The script automatically creates a private Python environment, installs the dependencies, creates a local `.env` file, and starts the app. When the terminal says it is running, open **http://127.0.0.1:5000**.

Keep that terminal open while using the app. Press **Ctrl+C** in it to stop the server. Run `bash setup.sh` again to start it later; already-installed packages are reused.

## Open in VS Code

1. Open the `AGROSAGE` folder in VS Code.
2. Press **Ctrl+Shift+B** and choose **AgriSage: Run (automatic setup)** if asked.
3. Open **http://127.0.0.1:5000** in your browser.

The VS Code task uses the same setup script. No Docker is needed for local development.

## Optional AI key

The app has built-in fallback guidance and can run without an API key. To enable optional Gemini image review, open the local `.env` file and set `GEMINI_API_KEY` to your own key. Never commit or share `.env`.

## Data and model

- The trained model and matching label files are included in the repository.
- Local prediction history and cached disease guidance are stored in `data/predictions.db`.
- Uploaded images are stored locally under `static/uploads/`.
- Keep important decisions checked by a qualified local agriculture expert; image predictions can be wrong.

## Troubleshooting

- **Python not found:** install Python 3.11 or 3.12 and reopen the terminal.
- **Package installation fails:** check the internet connection, then run `bash setup.sh` again.
- **Port 5000 is busy:** stop the other local server, or set a different `PORT` in `.env` before starting.
