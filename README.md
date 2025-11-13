This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## ML & Model placement (KILOS MVP notes)

This project includes early client-side modules for a browser ML triage flow used by the KILOS MVP.

1. Place the provided TensorFlow.js model under `public/models/hazard-tiny/` so the model manifest is available at:

	`/public/models/hazard-tiny/model.json`

2. Recommended npm packages to install locally (dev machine / before running):

	```bash
	npm install @tensorflow/tfjs leaflet react-leaflet supercluster heatmap.js blockhash-js
	```

	- `@tensorflow/tfjs` — browser inference runtime
	- `leaflet` + `react-leaflet` — interactive maps
	- `supercluster` — marker clustering for dense reports
	- `heatmap.js` — heat layer visualization (or react-leaflet-heatmap if preferred)
	- `blockhash-js` — perceptual hashing for duplicate detection (optional)

3. Files added:

	- `lib/HazardClassifier.ts` — dynamic TF.js loader + image/file classification helpers
	- `lib/priority.ts` — priority calculation and nearby verification boost

4. Next steps I can take for you:

	- Implement a mobile-first `app/report` form with camera upload and GPS accuracy validation (<50m).
	- Add a Leaflet-based heatmap and marker cluster component wired to a simulated WebSocket feed.
	- Wire the classifier into the report flow and apply priority logic (and duplicate detection option).

If you want me to continue now, tell me whether you want me to install the npm packages and implement the report page and map next (I can add components and wire them). I can also run a dev server locally in this workspace to validate build if you want.
