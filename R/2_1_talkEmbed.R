
#' Transform audio recordings to embeddings
#'
#' @param talk_filepaths (string) path to a video file (.wav/) list of audio filepaths, each is embedded separately
#' @param model shortcut name for Hugging Face pretained model. Full list https://huggingface.co/transformers/pretrained_models.html
#' @param audio_transcriptions  (strings) audio_transcriptions : list
#' (optional) list of audio transcriptions, to be used for Whisper's decoder-based embeddings
#' @param use_decoder (boolean) whether to use Whisper's decoder last hidden state representation
#' (Note: audio_transcriptions must be provided if this option is set to true)
#' @param tokenizer_parallelism  (boolean) whether to use device parallelization during tokenization.
#' @param device (string) name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
#' @param model_max_length (integer) maximum length of the tokenized text
#' @param hg_gated (boolean) set to True if the model is gated
#' @param hg_token (string) the token to access the gated model got in huggingface website
#' @param trust_remote_code (boolean) use a model with custom code on the Huggingface Hub.
#' @param logging_level (string) Set logging level, options: "critical", "error", "warning", "info", "debug".
#' @return A tibble with embeddings.
#' @examples
#' # Transform audio recordings in the example dataset:
#' # voice_data (included in talk-package), to embeddings.
#' \dontrun{
#'
#' }
#'
#' @seealso \code{\link{talkText}}.
#' @importFrom reticulate source_python
#' @importFrom tibble as_tibble
#' @export
talkEmbed <- function(
    talk_filepaths,
    model = "openai/whisper-small",
    audio_transcriptions = "None",
    use_decoder = FALSE,
    tokenizer_parallelism = FALSE,
    model_max_length = "None",
    device = 'cpu',
    hg_gated = FALSE,
    hg_token = "",
    trust_remote_code = FALSE,
    logging_level = 'warning'){



  reticulate::source_python(system.file("python",
                                        "huggingface_Interface4.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))


  embeddings <- hgTransformerGetEmbedding(
    audio_filepaths = talk_filepaths,
    audio_transcriptions = audio_transcriptions,
    model = model,
    use_decoder = use_decoder,
    tokenizer_parallelism = tokenizer_parallelism,
    model_max_length = model_max_length,
    device = device,
    hg_gated = hg_gated,
    hg_token = hg_token,
    trust_remote_code = trust_remote_code,
    logging_level = logging_level
  )

  embeddings <- embeddings[[1]]

  emb_tibble <- tibble::as_tibble(
    t(embeddings), # Transpose the vector into a single-row matrix
    .name_repair = ~ paste0("Dim", seq_along(embeddings)) # Assign column names
  )

  return(emb_tibble)
}


