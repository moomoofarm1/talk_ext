
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- -->

# talk <a href="https://r-talk.org"><img src="man/figures/logo.png" align="right" height="138" alt="talk website" /></a>

<!-- badges: start -->

[![Github build
status](https://github.com/theharmonylab/talk/workflows/R-CMD-check/badge.svg)](https://github.com/theharmonylab/talk/actions)
[![codecov](https://codecov.io/gh/theharmonylab/talk/branch/main/graph/badge.svg?)](https://app.codecov.io/gh/theharmonylab/talk)

<!--
[![CRAN Status](https://www.r-pkg.org/badges/version/talk)](https://CRAN.R-project.org/package=talk)
&#10;[![Lifecycle: maturing](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://lifecycle.r-lib.org/articles/stages.html#maturing-1)
&#10;[![CRAN Downloads](https://cranlogs.r-pkg.org/badges/grand-total/talk)](https://CRAN.R-project.org/package=talk)
&#10;-->

<!-- badges: end -->

## Overview

An R-package for analyzing natural language with transformers-based
large language models. The `talk` package is part of the *R Language
Analysis Suite*, including `talk`, `text` and `topics`.

- [`talk`](https://www.r-talk.org/) transforms voice recordings into
  text, audio features, or embeddings.<br> <br>
- [`text`](https://www.r-text.org/) provide many language tasks such as
  converting digital text into word embeddings.<br> <br> `talk` and
  `text` provide access to Large Language Models from Hugging Face.<br>
  <br>
- [`topics`](https://www.r-topics.org/) visualizes language patterns
  into topics to generate psychological insights.<br> <br> <br>
  <img src="man/figures/talk_text_topics1.svg" style="width:50.0%" />

<br> The *R Language Analysis Suite* is created through a collaboration
between psychology and computer science to address research needs and
ensure state-of-the-art techniques. The suite is continuously tested on
Ubuntu, Mac OS and Windows using the latest stable R version.

### Short installation guide

Most users simply need to run below installation code. For those
experiencing problems, please see the [Extended Installation
Guide](https://www.r-talk.org/articles/huggingface_in_r_extended_installation_guide.html).

For the talk-package to work, you first have to install the talk-package
in R, and then make it work with talk required python packages.

1.  Install talk-version (at the moment the second step only works using
    the development version of talk from GitHub).

[GitHub](https://github.com/) development version:

``` r
# install.packages("devtools")
devtools::install_github("theharmonylab/talk")
```

<!--
[CRAN](https://CRAN.R-project.org/package=talk) version:
&#10;``` r
install.packages("talk")
```
-->

2.  Install and initialize talk required python packages:

``` r
library(talk)

# Install talk required python packages in a conda environment (with defaults).
talkrpp_install()

# Initialize the installed conda environment.
# save_profile = TRUE saves the settings so that you don't have to run talkrpp_initialize() after restarting R. 
talkrpp_initialize(save_profile = TRUE)
```

### Point solution for transforming talk to embeddings

Recent significant advances in NLP research have resulted in improved
representations of human language (i.e., language models). These
language models have produced big performance gains in tasks related to
understanding human language. talk are making these SOTA models easily
accessible through an interface to
[HuggingFace](https://huggingface.co/docs/transformers/index) in Python.

See [HuggingFace](https://huggingface.co/models/) for a more
comprehensive list of models.

The `talkText()` function performs speech-to-text, transcribing audio
input to text. `talkEmbed()`, transforms audio input to numeric
representations (embeddings) that can be used for downstream tasks such
as guideline predictive models using the text-package (see the text
train functions).

``` r
library(talk)
# Transform the talk data to BERT word embeddings

# Get file path to example audio from the package example data
wav_path <- system.file("extdata/",
                            "test_short.wav",
                            package = "talk")

# Get transcription 
talk_embeddings <- talkText(
  wav_path
)
talk_embeddings

# Defaults
talk_embeddings <- talkEmbed(
  wav_path
)
talk_embeddings
```
