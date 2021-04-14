# COG403Project
Python codebase for U of T COG403 Winter 2021 final project.

This codebase contains a tweaked version of the AO3Scraper library (https://github.com/radiolarian/AO3Scraper), as well as tweaked versions of several scripts from the toastystats library (https://github.com/fandomstats/toastystats) and code from mbednarski's PyTorch word2vec implementation (https://gist.github.com/mbednarski/da08eb297304f7a66a3840e857e060a0).

This codebase has the following dependencies:
* bs4
* lxml
* requests
* unidecode
* urllib3
* spacy
* torch
* numpy

On the subject of fandom selection: "Miraculous Ladybug" appears in both the "Cartoons & Comics & Graphic Novels" and "Anime & Manga" sections. It is a French-Italian-Japanese-South Korean joint production, and French, English, Korean, and Japanese are all original languages for it. For simplicity's sake, it was excluded from both categories in our analysis.

As part of its output, train_model.py generates PyTorch parameter files (.pt) to store the parameters of the trained models after each training epoch. These files were over 20 MB each, and there were 10 of them (2 models x 5 epochs), so the gitignore excludes them from the repo.
