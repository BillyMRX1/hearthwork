// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

// Where the site is served. Defaults are for the GitHub Pages project site.
// For a custom domain, build with SITE_URL=https://example.com SITE_BASE=/ (see docs-site/README.md).
const site = process.env.SITE_URL || 'https://billymrx1.github.io';
const rawBase = process.env.SITE_BASE ?? '/hearthwork';
const base = rawBase === '/' || rawBase === '' ? '' : '/' + rawBase.replace(/^\/+|\/+$/g, '');

// Markdown links and images that start with "/" are written without the base path;
// this adds it, so the same content works under /hearthwork and under a custom domain.
function remarkPrefixBase() {
	const walk = (node) => {
		if ((node.type === 'link' || node.type === 'image') && typeof node.url === 'string' && /^\/(?!\/)/.test(node.url)) {
			node.url = base + node.url;
		}
		if (node.children) node.children.forEach(walk);
	};
	return (tree) => walk(tree);
}

export default defineConfig({
	site,
	base: base || undefined,
	markdown: { remarkPlugins: [remarkPrefixBase] },
	integrations: [
		starlight({
			title: 'Hearthwork',
			description: 'Run a local AI model on your own computer and connect your coding agent to it.',
			logo: { src: './src/assets/flame.svg', alt: 'Hearthwork' },
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/BillyMRX1/hearthwork' }],
			editLink: { baseUrl: 'https://github.com/BillyMRX1/hearthwork/edit/main/docs-site/' },
			customCss: ['./src/styles/custom.css'],
			plugins: [starlightLinksValidator()],
			sidebar: [
				{
					label: 'Get started',
					items: [
						{ slug: 'get-started/check-your-pc' },
						{ slug: 'get-started/install' },
						{ slug: 'get-started/first-model' },
						{ slug: 'get-started/first-session' },
					],
				},
				{
					label: 'Guides',
					items: [
						{
							label: 'Agents',
							items: [
								{ slug: 'guides/claude-code' },
								{ slug: 'guides/codex' },
								{ slug: 'guides/opencode' },
								{ slug: 'guides/qwen-code' },
								{ slug: 'guides/aider' },
								{ slug: 'guides/aliases' },
							],
						},
						{ slug: 'guides/other-computer' },
						{ slug: 'guides/tailscale' },
						{ slug: 'guides/api-mode' },
						{ slug: 'guides/subagents' },
						{ slug: 'guides/budgets' },
						{ slug: 'guides/setup-profiles' },
						{ slug: 'guides/auto-tuning' },
						{ slug: 'guides/model-switching' },
						{ slug: 'guides/recovery' },
						{ slug: 'guides/benchmark' },
						{ slug: 'guides/mobile-app' },
					],
				},
				{
					label: 'Troubleshooting',
					items: [
						{ slug: 'troubleshooting' },
						{ slug: 'troubleshooting/doctor' },
						{ slug: 'troubleshooting/diagnose' },
						{ slug: 'troubleshooting/logs' },
						{ slug: 'troubleshooting/common-errors' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ slug: 'reference/commands' },
						{ slug: 'reference/cli' },
						{ slug: 'reference/data-and-config' },
						{ slug: 'reference/ports-and-environment' },
					],
				},
				{
					label: 'How it works',
					items: [
						{ slug: 'how-it-works' },
						{ slug: 'how-it-works/relay' },
						{ slug: 'how-it-works/context-and-models' },
					],
				},
				{
					label: 'Contributing & maintainers',
					items: [
						{ slug: 'contributing' },
						{ slug: 'contributing/tests' },
						{ slug: 'contributing/adding-an-agent' },
						{ slug: 'contributing/release' },
						{ slug: 'contributing/website' },
					],
				},
			],
		}),
	],
});
