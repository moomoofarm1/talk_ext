
#' Transform audio recordings to embeddings
#'
#' @param talk_filepaths (string) Path to a video file (.wav/) list of audio filepaths, each is embedded separately
#' @param model shortcut name for Hugging Face pretained model. Full list https://huggingface.co/transformers/pretrained_models.html
#' @param layers (string or list)  'all' or an integer list of layers to keep
#' @param return_tokens add
#' @param device add
#' @param tokenizer_parallelism  add
#' @param model_max_length add
#' @param hg_gated (boolean) Set to True if the model is gated
#' @param hg_token (string) The token to access the gated model got in huggingface website
#' @param trust_remote_code add
#' @param logging_level add
#' @param sentence_tokenize add
#' @param model (character) Path or name of the model to use for embedding the audio data.
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
#' @export
talkEmbed <- function(
    talk_filepaths,
    model = 'whisper-tiny',
    layers = 'all',
    return_tokens = TRUE,
    device = 'cpu',
    tokenizer_parallelism = FALSE,
    model_max_length = None,
    hg_gated = FALSE,
    hg_token = "",
    trust_remote_code = FALSE,
    logging_level = 'warning',
    sentence_tokenize = TRUE){



  reticulate::source_python(system.file("python",
                                        "huggingface_Interface4.py",
                                        package = "talk",
                                        mustWork = TRUE
  ))


  output <- get_audio_embeddings(
    audio_filepaths = talk_filepaths,
    model = model,
    layers = layers,
    return_tokens = return_tokens,
    device = device,
    tokenizer_parallelism = tokenizer_parallelism,
   # model_max_length = model_max_length,
    hg_gated = hg_gated,
    hg_token = hg_token,
    trust_remote_code = trust_remote_code,
    logging_level = logging_level,
    sentence_tokenize = sentence_tokenize
  )

  return(output)


}


