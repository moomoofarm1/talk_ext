#' @importFrom utils packageVersion
#' @noRd
.onAttach <- function(libname, pkgname) {
  if (!grepl(x = R.Version()$arch, pattern = "64")) {
    warning("The talk-package requires running R on a 64-bit systems
            as it is dependent on torch from ptyhon; and you're
            system is not 64-bit.")
  }

  talk_version_nr <- tryCatch(
    {
      talk_version_nr1 <- paste(" (version ", packageVersion("talk"), ")", sep = "")
    },
    error = function(e) {
      talk_version_nr1 <- ""
    }
  )

  OMP_msg <- ""
  if (Sys.info()[["sysname"]] == "Darwin") {
    skip_patch <- Sys.getenv("TEXT_SKIP_OMP_PATCH", unset = "FALSE")
    if (tolower(skip_patch) != "true") {
      Sys.setenv(OMP_NUM_THREADS = "1")
      Sys.setenv(OMP_MAX_ACTIVE_LEVELS = "1")
      Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")
      OMP_msg <- c("\n MacOS detected: Setting OpenMP environment variables to avoid potential crash due to libomp conflicts. \n")
    } else {
      OMP_msg <- c("Skipped setting OpenMP environment variables (TEXT_SKIP_OMP_PATCH is TRUE)")
    }
  }

  nowarranty <- c("The topics package is provided 'as is' without any warranty of any kind. \n")


  packageStartupMessage(
    colourise(
      paste("This is talk.",
            talk_version_nr,
            ".\n",
            sep = ""
      ),
      fg = "blue", bg = NULL
    ),
    colourise("Newer versions may have improved functions and updated defaults to reflect current understandings of the state-of-the-art.",
              fg = "green", bg = NULL
    ),
    colourise(OMP_msg,
              fg = "purple", bg = NULL
    ),
    colourise(nowarranty,
              fg = "purple", bg = NULL
    ),
    colourise("\n\nFor more information about the package see www.r-talk.org.",
              fg = "green", bg = NULL
    )
  )

  if (isTRUE(check_talkrpp_python_options()$val == "talkrpp_condaenv")) {
    talkrpp_initialize(check_env = FALSE)
  }
}

# Below function is from testthat:
# https://github.com/r-lib/testthat/blob/717b02164def5c1f027d3a20b889dae35428b6d7/R/colour-talk.r
#' Colourise talk for display in the terminal.
#'
#' If R is not currently running in a system that supports terminal colours
#' the talk will be returned unchanged.
#'
#' Allowed colours are: black, blue, brown, cyan, dark gray, green, light
#' blue, light cyan, light gray, light green, light purple, light red,
#' purple, red, white, yellow
#'
#' @param talk character vector
#' @param fg foreground colour, defaults to white
#' @param bg background colour, defaults to transparent
# @examples
#' @noRd
colourise <- function(talk, fg = "black", bg = NULL) {
  term <- Sys.getenv()["TERM"]
  colour_terms <- c("xterm-color", "xterm-256color", "screen", "screen-256color")

  if (rcmd_running() || !any(term %in% colour_terms, na.rm = TRUE)) {
    return(talk)
  }

  col_escape <- function(col) {
    paste0("\033[", col, "m")
  }

  col <- .fg_colours[tolower(fg)]
  if (!is.null(bg)) {
    col <- paste0(col, .bg_colours[tolower(bg)], sep = ";")
  }

  init <- col_escape(col)
  reset <- col_escape("0")
  paste0(init, talk, reset)
}

.fg_colours <- c(
  "black" = "0;30",
  "blue" = "0;34",
  "green" = "0;32",
  "cyan" = "0;36",
  "red" = "0;31",
  "purple" = "0;35",
  "brown" = "0;33"
  # "light gray" = "0;37",
  # "dark gray" = "1;30",
  # "light blue" = "1;34",
  # "light green" = "1;32",
  # "light cyan" = "1;36",
  # "light red" = "1;31",
  # "light purple" = "1;35",
  # "yellow" = "1;33",
  # "white" = "1;37"
)

.bg_colours <- c(
  "black" = "40",
  "red" = "41",
  "green" = "42",
  "brown" = "43",
  "blue" = "44",
  "purple" = "45",
  "cyan" = "46",
  "light gray" = "47"
)

rcmd_running <- function() {
  nchar(Sys.getenv("R_TESTS")) != 0
}

