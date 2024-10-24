
<p align="center">
  <img src="https://i.imgur.com/5LJxBt4.pngg" height="350" alt="AgentGPT Logo"/>
</p>
<p align="center">
  <em>🩺 Healthcare at your fingertips! 🖐   </em>
</p>

[MediScan](https://mediscan.tech) gives you easy access to wait times for hospitals nearby and allows you to self diagnose so you can have the best medical experience 🚀!
The app is split into two main categories: Giving hospital wait times and predicting injuries or diseases through a convolutional neural network (CNN) model.

Source Code for Website: https://github.com/mediscan-tech/www
---
## About our Data:
Our data is completely dynamic. The dataset from [CMS](https://data.cms.gov/) is updated every few weeks, so our hospital wait times are updated automatically.
---
## ✨ Demo
For the best demo experience, visit [our site](https://mediscan.tech) :)

[Demo Video](https://www.youtube.com/watch?v=ANMwVpiqOHk)

## 🚀 Website Tech Stack
- ✅ **Bootstrapping**: [create-t3-app](https://create.t3.gg).
- ✅ **Framework**: [NextJS 14](https://nextjs.org/) + [Typescript](https://www.typescriptlang.org/).
- ✅ **Styling**: [TailwindCSS](https://tailwindcss.com) + [RadixUI](https://www.radix-ui.com/).
- ✅ **Animations**: [AOS](https://michalsnik.github.io/aos/) + [HeadlessUI](https://headlessui.com/).
- ✅ **Component Library**: [shadcn/ui](https://ui.shadcn.com/).
- ✅ **Initial Landing Page Template**: [Cruip](https://cruip.com/).
- ✅ **Database**: [MongoDB](https://www.mongodb.com/).
- ✅ **Schema Validation**: [Zod](https://zod.dev/).
- ✅ **File Uploads**: [Uploadthing](https://uploadthing.com/).
- ✅ **Data Caching**: [Redis](https://redis.com/) + [Upstash](https://upstash.com/).
- ✅ **Medical Data**: [Centers for Medicare and Medicaid Services](https://data.cms.gov/).
- ✅ **Reverse Geocoding APIs**: [Opencage](https://opencagedata.com/) + [Geonames](https://www.geonames.org/export/web-services.html).
- ✅ **Geocoding APIs**: [Google Maps](https://developers.google.com/maps) + [MapBox Geocoding](https://docs.mapbox.com/api/search/geocoding/).
- ✅ **Map Renderer**: [MapBox Maps](https://www.mapbox.com/maps).
- ✅ **Hosting**: [Vercel](https://vercel.com/).

## 🤖 AI Model Tech Stack
- ✅ **Language**: [Python](https://www.python.org/).
- ✅ **ML Library**: [Tensorflow](https://www.tensorflow.org/).
- ✅ **Datasets**: [Kaggle](https://www.kaggle.com/).
- ✅ **Framework**: [Flask](https://flask.palletsprojects.com/en/2.0.x/) + [Gunicorn](https://gunicorn.org/).
- ✅ **Hosting**: [DigitalOcean](https://www.digitalocean.com/).

## Digital Ocean Hosting Tips to Host the 3 Models
- Command to run the app platform: `gunicorn --worker-tmp-dir /dev/shm --timeout 120 app:app`
  - Used to host the flask api that handles image inputs for all models
  - Has a larger timeout to run the models
- Full Digital Ocean App Platform Specification: https://gist.github.com/navincodesalot/1e74f2f1ffe3bd22cdc5ab9f1d1de645
- Host model on Digital Ocean Spaces Object Storage (like AWS S3) so it can download via URL and save hardware space through git commits and etc.
- setup.py Downloads the Larger Model
- All Models Trained on Python Version 3.9.7 and Tensorflow 2.13.0

---
## 🙌 Contributors 
<a href="https://github.com/navincodesalot"><img height="128" src="https://avatars.githubusercontent.com/u/67123306?v=4"/></a>
<a href="https://github.com/Richp16"> <img height="128" src="https://avatars.githubusercontent.com/u/67066931?v=4"/></a>
<a href="https://github.com/Viverino1"><img height="128" src="https://avatars.githubusercontent.com/u/82279322?v=4"></a>
<a href="https://github.com/RishiMadhavan-ui"><img height="128" src="https://avatars.githubusercontent.com/u/86448548?v=4"></a>
