"""Pinned recorded identities and their Git revisions after 2026-11-04.

The public history was rewritten: private paths, internal hosts and identities
were replaced. Existing run/checkpoint metadata still records the old SHAs.
Keep those identities unchanged; translate only at Git-operation boundaries.
This is an exact pin map, not permission to bypass ancestry or content checks.
"""

SMALL40_RECORDED_SHA = "00cee786df783f36cb254a9a666a48a5952dc0a3"
H3_RECORDED_SHA = "e835fe39f6d208272a28afb3555f408854e14ca2"

RECORDED_TO_GIT_SHA = {
    SMALL40_RECORDED_SHA: "2f995fda636e0b5c8c0b7d50c37ba62f5bbc4e82",
    H3_RECORDED_SHA: "a8a6cd7a92ccfb076a23c14919c75f12c83e92fd",
}


def git_revision(revision: str) -> str:
    """Translate an exact pinned SHA or SHA:path; leave other Git args alone.

    Do not use this for owner/receipt/CLI identity comparisons. Short SHAs and
    unknown revisions are deliberately not guessed or resolved via old objects.
    """
    sha, separator, path = revision.partition(":")
    return RECORDED_TO_GIT_SHA.get(sha, sha) + separator + path
