


library(talk)
library(tibble)
library(testthat)

context("Getting embedding")


test_that("textEmbed handling NAs", {
  #skip_on_cran()

  # Install Required Backend:
  # •	Install the ffmpeg backend, which is often required for torchaudio to handle various audio formats:


  # Error: openssl@1.1 has been disabled because it is not supported upstream! It was disabled on 2024-10-24.

  #Ensure that ffmpeg is also installed on your system:
  #  •	On macOS: brew install ffmpeg
  #  •	On Ubuntu: sudo apt-get install ffmpeg
  #  •	On Windows: Use a precompiled binary from the FFmpeg website.

  whisp_embeddings <- talk::talkEmbed(
    talk_filepaths = "/Users/oscarkjell/Desktop/1 Projects/0 Research/0_talk/oscar_victoria.wav",
    model = 'openai/whisper-small',
    model_max_length = NULL
  )



})
