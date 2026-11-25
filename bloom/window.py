"""Pygame front end: a full-screen (or windowed) interactive view of the bouquet."""

from __future__ import annotations

import os
import time as _time

import numpy as np

from .draw import BG, GlyphCache, load_font, pygame
from .genome import random_seed
from .glitter import Glitter
from .scene import Scene


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
    clock = pygame.time.Clock()
    glitter = Glitter()

    dragging = False
    drag_moved = 0

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
                elif event.key == pygame.K_p:
                    scene.next_palette()
                elif event.key == pygame.K_s:
                    os.makedirs("screenshots", exist_ok=True)
                    path = os.path.join("screenshots", f"bloom-{scene.seed}-{int(_time.time())}.png")
                    pygame.image.save(screen, path)
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
        screen.fill(BG)
        glitter.draw(screen, scene.time, dt, frame, (glyphs.cw, glyphs.ch), glyphs.origin(screen, frame))
        glyphs.draw(screen, frame, clear=False)
        scene.draw_overlay(screen)  # the only text on screen: the words under the bouquet
        pygame.display.flip()

    pygame.quit()
