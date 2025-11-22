# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Prerequisites

- **Node.js** version 20.0 or higher ([install via nodejs.org](https://nodejs.org/) or [nvm](https://github.com/nvm-sh/nvm))
- **npm** (comes with Node.js) or **yarn** (install via `npm install -g yarn`)

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=your_github_username yarn deploy
```

**Note:** Replace `your_github_username` with your actual GitHub username (not the literal text). If prompted for authentication, you may also need to provide a personal access token via the `GITHUB_TOKEN` environment variable:

```bash
GIT_USER=your_github_username GITHUB_TOKEN=your_token yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
