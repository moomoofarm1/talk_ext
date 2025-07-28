

#' Transcribe and Diarize Audio Recordings
#'
#' This function transcribes one or more audio recordings using Whisper and optionally performs speaker diarization using NeMo.
#' It is intended for converting speech audio into text and, if enabled, assigning speaker labels to segments.
#'
#' @param audio (string or list) Path to a single audio file (e.g., `.wav`) or a list of file paths. Each file is processed separately.
#' @param output_dir (string) Directory where output files (transcripts, speaker-labeled files) will be saved. If NULL, uses a temp directory.
#' @param stemming (logical) If TRUE, words will be stemmed in the output.
#' @param suppress_numerals (logical) If TRUE, numerals will be removed from the transcript.
#' @param model (string) Name of the Whisper model to use (e.g., `"medium.en"`). Available models include `tiny`, `base`, `small`, `medium`, `large`.
#' @param batch_size (integer) Number of audio segments to process at once. Useful for large files or GPU acceleration. Default is 8.
#' @param max_retries (integer) Maximum number of retries for failed decoding attempts. Default is 1.
#' @param max_oom_retries (integer) Maximum number of retries if a CUDA out-of-memory error occurs.
#' @param language (string) Language code (e.g., `"en"`, `"sv"`). If NULL, Whisper will attempt to detect the language.
#' @param rerun_completed_audio_file (logical) If TRUE, forces rerun even if output already exists.
#' @param diarize_worker (logical) If TRUE, enables speaker diarization using NeMo. Requires NeMo and related dependencies.
#' @param temp_dir (string) Optional path to temporary directory for intermediate files.
#'
#' @return Nothing returned. Output files (e.g., transcripts and diarization CSVs) are written to `output_dir`.
#'
#' @details
#' The function internally uses Python libraries through `reticulate`. It selects `"cuda"` if a GPU is available and properly configured;
#' otherwise it defaults to `"cpu"`. You can retrieve and parse output files manually from the specified `output_dir`.
#'
#' @examples
#' \dontrun{
#' wav_path <- system.file("extdata", "test_short.wav", package = "talk")
#' talk::talkTranscribeDiarise(audio = wav_path)
#' }
#'
#' @seealso \code{\link{talkEmbed}}, \code{\link{talkText}}, \code{\link{talkrpp_initialize}}
#' @importFrom reticulate source_python py_module_available import
#' @importFrom tibble as_tibble
#' @export
talkTranscribeDiarise <- function(
    audio = NULL,
    output_dir = getwd(),
    model = "medium.en",
    stemming = TRUE,
    suppress_numerals = FALSE,
    batch_size = 8,
    max_retries = 1,
    max_oom_retries = 2,
    language = NULL,
    rerun_completed_audio_file = FALSE,
    diarize_worker = TRUE,
    temp_dir = getwd()
    ){

  reticulate::source_python(system.file("python",
                                        "whisnemo_diarization.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))

  # Determine device (CUDA vs CPU)
  device <- "cpu"
  if (reticulate::py_module_available("torch")) {

    torch <- reticulate::import("torch")
    if (torch$cuda$is_available()) {
      device <- "cuda"
    }
  }


  # Call the Python function
  transcribe_and_diarize(
    audio = audio,
    output_dir = output_dir,
    stemming = stemming,
    suppress_numerals = suppress_numerals,
    model_name = "medium",
    batch_size = batch_size,
    max_retries = max_retries,
    max_oom_retries = max_oom_retries,
    language = language,
    device = device,
    rerun_completed_audio_file = rerun_completed_audio_file,
    diarize_worker = FALSE, #diarize_worker,
    temp_dir = "/Users/oscarkjell/Desktop/1 Projects/0 Research/0_talk"
  )

#  embeddings <- embeddings[[1]]
#
#  emb_tibble <- tibble::as_tibble(
#    t(embeddings), # Transpose the vector into a single-row matrix
#    .name_repair = ~ paste0("Dim", seq_along(embeddings)) # Assign column names
#  )

  return(emb_tibble)
}


