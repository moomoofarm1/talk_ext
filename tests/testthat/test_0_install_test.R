library(testthat)
library(talk)

test_that("installing talk", {
  skip_on_cran()

  # On Linux get error at reticulate:::pip_install(...)

#  if (Sys.info()["sysname"] == "Darwin" | Sys.info()["sysname"] == "Windows") {
    talk::talkrpp_install(prompt = FALSE,
                          envname = "talk_test"
                          )

    talk::talkrpp_initialize(talkEmbed_test = TRUE,
                             save_profile = FALSE,
                             prompt = FALSE,
                             condaenv = "talk_test"
                             )

    wav_path <- system.file("extdata/",
                            "test_short.wav",
                            package = "talk")
    wav_path

    emb_test <- talk::talkEmbed(
      talk_filepaths = wav_path,
      model = "openai/whisper-tiny"
    )

    testthat::expect_equal(emb_test$Dim1,
                           -0.2030126, tolerance = 0.0001)
    testthat::expect_equal(emb_test$Dim2,
                           -1.008844, tolerance = 0.0001)
    testthat::expect_equal(emb_test$Dim3,
                           0.897202, tolerance = 0.0001)
#  }

  text_test <- talk::talkText(
    talk_filepaths = wav_path,
    model = "openai/whisper-tiny"
  )

  testthat::expect_equal(text_test[1],
                         " Hello.")

#    INSTEAD SEE HOW IT IS BEING UNINSTALLED IN TEXT_ZZ... file
    if (Sys.info()["sysname"] == "Darwin" | Sys.info()["sysname"] == "Windows") {

#    # help(textrpp_uninstall)
#    text::textrpp_install(prompt = FALSE,
#                          envname = "uninstall")

    talkrpp_uninstall(prompt = FALSE,
                      envname = "talk_test")
  }

})
