"""Inferix eval/preview/profiling backend placeholder."""

from __future__ import annotations


class InferixEvalBackend:
    def generate_preview(self, *args, **kwargs):
        raise NotImplementedError("Inferix preview backend is planned for v0.4.")

    def profile_checkpoint(self, *args, **kwargs):
        raise NotImplementedError("Inferix profiling backend is planned for v0.4.")

    def run_long_video_eval(self, *args, **kwargs):
        raise NotImplementedError("Inferix long-video eval backend is planned for v0.4.")

