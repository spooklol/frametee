#ifndef USER_INTERFACE_SHOT_FINDER_H
#define USER_INTERFACE_SHOT_FINDER_H

#include <stdbool.h>
#include <system/include_cimgui.h>

struct ui_handler_t;

// Built-in DDNet shot searcher. The implementation deliberately uses the
// editor's opaque game/world interfaces, so all movement still comes from the
// active game module's real simulation rather than a second physics model.
void shot_finder_init(struct ui_handler_t *ui);
void shot_finder_cleanup(struct ui_handler_t *ui);
void shot_finder_reset(struct ui_handler_t *ui, const char *status);
void shot_finder_update(struct ui_handler_t *ui);
void shot_finder_render_window(struct ui_handler_t *ui);
bool *shot_finder_window_visibility(struct ui_handler_t *ui);

// Draws and handles the direction picker over the viewport image. True means
// the left click was consumed and must not also select a player/start point.
bool shot_finder_viewport_overlay(struct ui_handler_t *ui, ImVec2 viewport_origin,
                                  ImVec2 viewport_size, bool hovered);

#endif // USER_INTERFACE_SHOT_FINDER_H
