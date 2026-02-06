# orlaserver.github.io

GitHub Pages site for [orlaserver](https://github.com/orlaserver).

## Setup

1. In the [orlaserver](https://github.com/orlaserver) org, create a new repository named **exactly** `orlaserver.github.io` (public).
2. Push this folder’s contents to the `main` branch:

   ```bash
   cd orlaserver.github.io
   git init
   git add .
   git commit -m "Initial site"
   git remote add origin https://github.com/orlaserver/orlaserver.github.io.git
   git branch -M main
   git push -u origin main
   ```

3. The site will be live at **https://orlaserver.github.io** (may take a minute or two).

## Editing

- **Homepage:** edit `index.html`.
- **Styles:** edit `styles.css`.
- Add more `.html` pages as needed and link to them from `index.html`.

No build step required; GitHub Pages serves the files as-is.
