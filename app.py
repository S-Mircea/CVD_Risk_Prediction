"""Vercel entrypoint: exposes the Flask app from app_code/web_app.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'app_code'))

from web_app import app  # noqa: E402

__all__ = ['app']
