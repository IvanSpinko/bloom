"""Pygame front end: a full-screen (or windowed) interactive view of the bouquet."""

from __future__ import annotations

import os
import time as _time

import numpy as np

from .draw import FONT_NAMES, GlyphCache, load_font, pygame
from .genome import random_seed
from .scene import Scene


HELP = [
    "drag  turn        wheel  zoom",
    "click  wind gust  space  replay",
    "n  new bouquet    p  palette",
    "f  fullscreen     s  screenshot",
    "h  hide help      esc  quit",
]


def _open(fullscreen: bool, size: tuple[int, int]) -> pygame.Surface:
    if fullscreen:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return pygame.display.set_mode(size, pygame.RESIZABLE)


def run(scene: Scene, width: int, height: int, font_size: int | None, fps: int, fullscreen: bool = True) -> None:
    pygame.init()
    pygame.display.set_caption("bloom")
    pygame.mouse.set_visible(not fullscreen)
    screen = _open(fullscreen, (width, height))
    # the glyph size follows the screen height unless it is given explicitly
    glyphs = GlyphCache(load_font(font_size or max(8, round(screen.get_height() / 90))))
    small = pygame.font.SysFont(FONT_NAMES, 14)
    clock = pygame.time.Clock()

    show_help = False
    dragging = False
    drag_moved = 0
    toast, toast_until = "", 0.0

    def notify(text: str) -> None:
        nonlocal toast, toast_until
        toast, toast_until = text, _time.monotonic() + 2.0

    notify("h  help")
    running = True
    while running:
        dt = min(clock.tick(fps) / 1000.0, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_f, pygame.K_F11):
                    fullscreen = not fullscreen
                    screen = _open(fullscreen, (width, height))
                    pygame.mouse.set_visible(not fullscreen)
                elif event.key == pygame.K_SPACE:
                    scene.replay()
                elif event.key == pygame.K_n:
                    scene.plant(random_seed())
                    notify(f"new bouquet: {scene.seed}")
                elif event.key == pygame.K_p:
                    notify(f"palette: {scene.next_palette()}")
                elif event.key == pygame.K_h:
                    show_help = not show_help
                    pygame.mouse.set_visible(show_help or not fullscreen)
                elif event.key == pygame.K_s:
                    os.makedirs("screenshots", exist_ok=True)
                    path = os.path.join("screenshots", f"bloom-{scene.seed}-{int(_time.time())}.png")
                    pygame.image.save(screen, path)
                    notify(f"saved {path}")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging, drag_moved = True, 0
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging and drag_moved < 6:
                    scene.blow()
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                drag_moved += abs(dx) + abs(dy)
                scene.yaw_offset -= dx * 0.01
                scene.camera.pitch = float(np.clip(scene.camera.pitch + dy * 0.01, -0.3, 1.45))
            elif event.type == pygame.MOUSEWHEEL:
                scene.camera.zoom = float(np.clip(scene.camera.zoom * (1.1 ** event.y), 0.4, 4.0))

        scene.step(dt)

        w, h = screen.get_size()
        frame = scene.frame(*glyphs.grid_for(w, h))
        glyphs.draw(screen, frame)
        scene.draw_overlay(screen)

        if show_help:
            y = 10
            for line in [f"seed  {scene.seed}   palette  {scene.palette}   {clock.get_fps():.0f} fps"] + HELP:
                screen.blit(small.render(line, True, (130, 120, 100)), (12, y))
                y += 18
        if toast and _time.monotonic() < toast_until:
            img = small.render(toast, True, (200, 185, 150))
            screen.blit(img, ((w - img.get_width()) // 2, h - 30))

        pygame.display.flip()

    pygame.quit()
