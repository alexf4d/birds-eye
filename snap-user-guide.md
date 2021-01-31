# birds-eye <!-- omit in toc -->
# Snap <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)

## 1. Introduction
Birds Eye Snap is a tool to scrub through live and recorded video, detect birds, and then save pictures.

## 2. Usage

1. Help:
   - `python3 snap.py -h`
2. Using a YouTube video:
   - `python3 snap.py -s <stream_URL>`
3. Using a local file:
   - `python3 snap.py -f <input_file_path>`

All images will be saved to the directory "/images" within the project directory. If the directory does not exist one will be created.

Images in the directory are overwriten with each run, so make sure to move your desired images to a safe location.
