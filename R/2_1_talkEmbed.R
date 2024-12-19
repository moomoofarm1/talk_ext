
#' Transform audio recordings to embeddings
#'
#' @param talk (audio file) .
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
    talk,
    model = "WhiSPA"){



  reticulate::source_python(system.file("python",
                                        "whispa_embeddings.py",
                                        package = "text",
                                        mustWork = TRUE
  ))






}


