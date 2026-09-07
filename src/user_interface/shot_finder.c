#include "shot_finder.h"

#include "timeline/timeline_commands.h"
#include "timeline/timeline_model.h"
#include "undo_redo.h"
#include "user_interface.h"
#include "widgets/imcol.h"
#include <GLFW/glfw3.h>
#include <engine/game_host.h>
#include <engine/input_record.h>
#include <float.h>
#include <math.h>
#include <renderer/graphics_backend.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
  SHOT_FINDER_COARSE_ANGLES = 180, // Complete circle in two-degree steps.
  SHOT_FINDER_REFINE_ANGLES = 41,  // +/- two degrees in 0.1-degree steps.
  // A shot on C - 1 may legitimately explode during the step into target C.
  // Projectile timing, rather than an arbitrary lead, decides if it is useful.
  SHOT_FINDER_EARLY_MIN_LEAD = 1,
  SHOT_FINDER_CURRENT_EFFECT_WINDOW = 5,
  // "At the selected frame" has a small tolerance for the tick on which DDNet
  // removes the projectile and applies its explosion impulse.
  SHOT_FINDER_TARGET_WINDOW = 3,
  SHOT_FINDER_MIN_FUTURE = 150,    // Long enough for a grenade to take effect.
  SHOT_FINDER_MAX_FUTURE = 2000,   // Forty seconds of DDNet future at 50 Hz.
};

static const double SHOT_FINDER_FRAME_BUDGET = 0.004;
static const double SHOT_FINDER_IMPROVEMENT_EPSILON = 0.015;
static const double SHOT_FINDER_DIRECTION_EPSILON = 0.01;
static const double SHOT_FINDER_SPEED_EPSILON = 0.0001;

typedef enum shot_finder_kind_t {
  SHOT_FINDER_CURRENT = 0,
  SHOT_FINDER_EARLY,
  SHOT_FINDER_MULTI_EARLY,
} shot_finder_kind_t;

typedef enum shot_finder_phase_t {
  SHOT_FINDER_IDLE = 0,
  SHOT_FINDER_PICKING,
  SHOT_FINDER_SCAN_EARLY,
  SHOT_FINDER_BUILD_BASELINE,
  SHOT_FINDER_SEARCH_COARSE,
  SHOT_FINDER_SEARCH_REFINE,
} shot_finder_phase_t;

typedef struct shot_sample_t {
  ft_vec2 position;
  ft_vec2 velocity;
  uint32_t flags;
  bool valid;
} shot_sample_t;

typedef struct candidate_score_t {
  double weighted_progress;
  double lateral_distance;
  double maximum_progress;
  double end_progress;
  double maximum_velocity;
  double impact_velocity;
  double target_velocity;
  int impact_tick;
  bool impact_near_target;
  bool effect_within_current_window;
  int samples;
  int lost_alive_samples;
} candidate_score_t;

typedef struct planned_shot_t {
  int fire_tick;
  int impact_tick;
  double angle;
  double impact_velocity;
} planned_shot_t;

typedef struct shot_finder_t {
  bool show_window;
  shot_finder_kind_t kind;
  shot_finder_phase_t phase;

  int track;
  int group;
  int player;
  int selected_global_tick;
  int selected_local_tick;
  int earliest_fire_tick;
  int latest_fire_tick;
  int evaluation_end_tick;
  int scan_tick;
  bool scan_found_ready;
  int group_start_offset;
  ft_vec2 selection_origin;
  ft_vec2 desired_direction;

  int target_field;
  int fire_field;
  int weapon_field;
  int active_weapon_prop;
  int has_grenade_prop;
  int reload_timer_prop;
  int attack_tick_prop;
  int selected_weapon;
  int grenade_weapon;
  int projectile_class;
  int projectile_type_prop;
  int projectile_owner_prop;
  int projectile_start_tick_prop;
  int projectile_lifespan_prop;
  int projectile_explosive_prop;

  shot_sample_t *baseline;
  int baseline_count;
  int baseline_tick;
  ft_world *baseline_world;
  ft_world *candidate_seed_world;
  ft_world *candidate_world;
  unsigned char *packed_inputs;
  int player_count;
  size_t input_size;

  bool candidate_active;
  int seed_tick;
  int candidate_fire_tick;
  int candidate_sim_tick;
  int candidate_projectile_start_tick;
  int candidate_projectile_lifespan;
  bool candidate_projectile_alive;
  double candidate_angle;
  input_record_t candidate_input;
  candidate_score_t candidate_score;

  int coarse_frame;
  int coarse_angle;
  int refine_angle;
  int refine_frame;
  double refine_center_angle;
  int completed_candidates;
  int total_candidates;

  bool have_best;
  double best_score;
  double best_maximum_progress;
  double best_maximum_velocity;
  double best_impact_velocity;
  double best_target_velocity;
  int best_impact_tick;
  double best_angle;
  int best_fire_tick;

  planned_shot_t *planned_shots;
  int planned_shot_count;
  int planned_shot_capacity;
  double planned_maximum_progress;
  double planned_target_velocity;

  char status[256];
} shot_finder_t;

static shot_finder_t *finder(ui_handler_t *ui) { return ui ? ui->shot_finder : NULL; }

static bool is_ddnet(const ui_handler_t *ui) {
  if (!ui || !ui->gfx_handler) return false;
  const char *id = game_host_active_id(&ui->gfx_handler->game_host);
  return id && strcmp(id, "ddnet") == 0;
}

static void set_status(shot_finder_t *state, const char *format, ...) {
  if (!state) return;
  va_list args;
  va_start(args, format);
  vsnprintf(state->status, sizeof(state->status), format, args);
  va_end(args);
}

static bool search_running(const shot_finder_t *state) {
  return state && state->phase >= SHOT_FINDER_SCAN_EARLY;
}

static bool searches_past(const shot_finder_t *state) {
  return state && (state->kind == SHOT_FINDER_EARLY || state->kind == SHOT_FINDER_MULTI_EARLY);
}

static int entity_class_index(game_host_t *host, const char *id) {
  if (!host || !id) return -1;
  const unsigned count = gh_entity_class_count(host);
  for (unsigned i = 0; i < count; ++i) {
    const ft_entity_class *entity_class = gh_entity_class(host, i);
    if (entity_class && entity_class->id && strcmp(entity_class->id, id) == 0) return (int)i;
  }
  return -1;
}

static int entity_property(game_host_t *host, int entity_class, const char *id) {
  if (!host || entity_class < 0 || !id) return -1;
  const ft_entity_class *description = gh_entity_class(host, (unsigned)entity_class);
  if (!description) return -1;
  for (uint32_t i = 0; i < description->prop_count; ++i)
    if (description->props[i].id && strcmp(description->props[i].id, id) == 0) return (int)i;
  return -1;
}

static int player_property(game_host_t *host, const char *id) {
  return entity_property(host, FT_ENTITY_CLASS_PLAYER, id);
}

static bool entity_int(game_host_t *host, const ft_world *world, int entity_class, int entity,
                       int property, int *out) {
  if (!out || entity_class < 0 || property < 0) return false;
  ft_value value;
  if (!gh_entity_prop_get(host, world, (unsigned)entity_class, entity, (unsigned)property, &value) ||
      value.kind != FT_VALUE_INT)
    return false;
  *out = (int)value.as.i;
  return true;
}

static bool entity_bool(game_host_t *host, const ft_world *world, int entity_class, int entity,
                        int property, bool *out) {
  if (!out || entity_class < 0 || property < 0) return false;
  ft_value value;
  if (!gh_entity_prop_get(host, world, (unsigned)entity_class, entity, (unsigned)property, &value) ||
      value.kind != FT_VALUE_BOOL)
    return false;
  *out = value.as.b;
  return true;
}

static bool player_int(game_host_t *host, const ft_world *world, int player, int property, int *out) {
  if (!out || property < 0) return false;
  ft_value value;
  if (!gh_entity_prop_get(host, world, FT_ENTITY_CLASS_PLAYER, player, (unsigned)property, &value) ||
      value.kind != FT_VALUE_INT)
    return false;
  *out = (int)value.as.i;
  return true;
}

static bool player_bool(game_host_t *host, const ft_world *world, int player, int property, bool *out) {
  if (!out || property < 0) return false;
  ft_value value;
  if (!gh_entity_prop_get(host, world, FT_ENTITY_CLASS_PLAYER, player, (unsigned)property, &value) ||
      value.kind != FT_VALUE_BOOL)
    return false;
  *out = value.as.b;
  return true;
}

static shot_sample_t sample_player(game_host_t *host, const ft_world *world, int player) {
  shot_sample_t sample = {0};
  ft_player_view view = {.struct_size = sizeof(view)};
  if (gh_world_player_view(host, world, player, &view)) {
    sample.position = view.position;
    sample.velocity = view.velocity;
    sample.flags = view.flags;
    sample.valid = true;
  }
  return sample;
}

static int track_last_tick(const player_track_t *track) {
  int last = -1;
  if (!track) return last;
  for (int i = 0; i < track->snippet_count; ++i) {
    const input_snippet_t *snippet = &track->snippets[i];
    if (snippet->is_active && snippet->input_count > 0 && snippet->end_tick - 1 > last)
      last = snippet->end_tick - 1;
  }
  for (int i = 0; i < track->recording_snippet_count; ++i) {
    const input_snippet_t *snippet = &track->recording_snippets[i];
    if (snippet->is_active && snippet->input_count > 0 && snippet->end_tick - 1 > last)
      last = snippet->end_tick - 1;
  }
  return last;
}

static int group_last_tick(const timeline_state_t *timeline, int group) {
  int last = -1;
  for (int track = 0; track < timeline->player_track_count; ++track)
    if (model_track_group_index(timeline, track) == group) {
      const int track_last = track_last_tick(&timeline->player_tracks[track]);
      if (track_last > last) last = track_last;
    }
  return last;
}

static void release_search(ui_handler_t *ui, bool keep_phase) {
  shot_finder_t *state = finder(ui);
  if (!state) return;
  game_host_t *host = ui->gfx_handler ? &ui->gfx_handler->game_host : NULL;
  if (host && game_host_ready(host)) {
    gh_world_destroy(host, state->baseline_world);
    gh_world_destroy(host, state->candidate_seed_world);
    gh_world_destroy(host, state->candidate_world);
  }
  state->baseline_world = NULL;
  state->candidate_seed_world = NULL;
  state->candidate_world = NULL;
  free(state->baseline);
  free(state->packed_inputs);
  free(state->planned_shots);
  state->baseline = NULL;
  state->packed_inputs = NULL;
  state->planned_shots = NULL;
  state->baseline_count = 0;
  state->planned_shot_count = 0;
  state->planned_shot_capacity = 0;
  state->candidate_active = false;
  if (!keep_phase) state->phase = SHOT_FINDER_IDLE;
}

void shot_finder_reset(ui_handler_t *ui, const char *status) {
  shot_finder_t *state = finder(ui);
  if (!state) return;
  release_search(ui, false);
  if (status) set_status(state, "%s", status);
}

void shot_finder_init(ui_handler_t *ui) {
  if (!ui) return;
  ui->shot_finder = calloc(1, sizeof(*ui->shot_finder));
  shot_finder_t *state = finder(ui);
  if (!state) return;
  state->show_window = true;
  state->phase = SHOT_FINDER_IDLE;
  state->seed_tick = -1;
  set_status(state, "Ready. Select a DDNet player and TAS frame.");
}

void shot_finder_cleanup(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  if (!state) return;
  release_search(ui, false);
  free(state);
  ui->shot_finder = NULL;
}

bool *shot_finder_window_visibility(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  return state ? &state->show_window : NULL;
}

static bool capture_selection(ui_handler_t *ui, shot_finder_kind_t kind) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;

  if (!is_ddnet(ui)) {
    set_status(state, "ShotFinder is available when DDNet is the active game.");
    return false;
  }
  if (!ui->gfx_handler->level) {
    set_status(state, "Load a DDNet map first.");
    return false;
  }
  if (timeline->recording) {
    set_status(state, "Stop recording before running a shot search.");
    return false;
  }
  const int track = timeline->selected_player_track_index;
  if (track < 0 || track >= timeline->player_track_count) {
    set_status(state, "Select a player track first.");
    return false;
  }
  const int group = model_track_group_index(timeline, track);
  const int player = model_group_local_track_index(timeline, track);
  if (group < 0 || player < 0) {
    set_status(state, "The selected track is not attached to a simulation group.");
    return false;
  }

  release_search(ui, false);
  state->kind = kind;
  state->track = track;
  state->group = group;
  state->player = player;
  state->group_start_offset = timeline->groups[group]->start_offset;
  state->selected_local_tick = model_group_playhead_tick(timeline, group);
  state->selected_global_tick = state->selected_local_tick + state->group_start_offset;
  timeline->is_playing = false;
  timeline->is_reversing = false;

  const ft_world *world = model_group_world_at_tick(timeline, group, state->selected_global_tick);
  const shot_sample_t sample = sample_player(host, world, player);
  if (!sample.valid) {
    set_status(state, "Could not read the selected player's world state.");
    return false;
  }
  state->selection_origin = sample.position;
  state->active_weapon_prop = player_property(host, "active_weapon");
  state->selected_weapon = -1;
  if (!player_int(host, world, player, state->active_weapon_prop, &state->selected_weapon) ||
      state->selected_weapon < 0) {
    set_status(state, "Could not read the equipped weapon at the selected frame.");
    return false;
  }
  state->phase = SHOT_FINDER_PICKING;
  set_status(state, "Click in the viewport to choose the desired travel direction. Esc cancels.");
  return true;
}

static bool resolve_search_schema(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  game_host_t *host = &ui->gfx_handler->game_host;
  state->target_field = game_input_field_index(host, "target");
  state->fire_field = game_input_field_index(host, "fire");
  state->weapon_field = game_input_field_index(host, "weapon");
  state->active_weapon_prop = player_property(host, "active_weapon");
  state->has_grenade_prop = player_property(host, "has_grenade");
  state->reload_timer_prop = player_property(host, "reload_timer");
  state->attack_tick_prop = player_property(host, "attack_tick");
  state->grenade_weapon = -1;
  state->projectile_class = entity_class_index(host, "projectile");
  state->projectile_type_prop = entity_property(host, state->projectile_class, "type");
  state->projectile_owner_prop = entity_property(host, state->projectile_class, "owner");
  state->projectile_start_tick_prop = entity_property(host, state->projectile_class, "start_tick");
  state->projectile_lifespan_prop = entity_property(host, state->projectile_class, "lifespan");
  state->projectile_explosive_prop = entity_property(host, state->projectile_class, "explosive");
  const ft_input_schema *schema = game_input_schema(host);
  if (schema && state->weapon_field >= 0 && (uint32_t)state->weapon_field < schema->field_count) {
    const ft_input_field *weapon = &schema->fields[state->weapon_field];
    for (uint32_t i = 0; weapon->enum_labels && i < weapon->enum_count; ++i)
      if (weapon->enum_labels[i] && strcmp(weapon->enum_labels[i], "Grenade") == 0)
        state->grenade_weapon = weapon->min_value + (int)i;
  }

  if (state->target_field < 0 || state->fire_field < 0 || state->weapon_field < 0 ||
      state->active_weapon_prop < 0 || state->attack_tick_prop < 0 || state->reload_timer_prop < 0) {
    set_status(state, "The active DDNet module does not expose the input/shot state ShotFinder needs.");
    state->phase = SHOT_FINDER_IDLE;
    return false;
  }
  if (searches_past(state) &&
      (state->grenade_weapon < 0 || state->has_grenade_prop < 0 || state->reload_timer_prop < 0 ||
       state->projectile_class < 0 || state->projectile_type_prop < 0 || state->projectile_owner_prop < 0 ||
       state->projectile_start_tick_prop < 0 || state->projectile_lifespan_prop < 0 ||
       state->projectile_explosive_prop < 0)) {
    set_status(state, "The active DDNet module does not expose grenade cooldown/projectile state.");
    state->phase = SHOT_FINDER_IDLE;
    return false;
  }
  return true;
}

static bool allocate_search(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;

  if (searches_past(state) &&
      (state->latest_fire_tick > state->selected_local_tick - SHOT_FINDER_EARLY_MIN_LEAD ||
       state->earliest_fire_tick > state->latest_fire_tick)) {
    set_status(state, "No legal grenade firing frame exists before the selected target frame.");
    state->phase = SHOT_FINDER_IDLE;
    return false;
  }

  const int authored_end = group_last_tick(timeline, state->group) + 1;
  int evaluation_end = authored_end;
  if (evaluation_end < state->selected_local_tick + SHOT_FINDER_MIN_FUTURE)
    evaluation_end = state->selected_local_tick + SHOT_FINDER_MIN_FUTURE;
  if (evaluation_end > state->selected_local_tick + SHOT_FINDER_MAX_FUTURE)
    evaluation_end = state->selected_local_tick + SHOT_FINDER_MAX_FUTURE;
  state->evaluation_end_tick = evaluation_end;
  state->baseline_count = evaluation_end - state->earliest_fire_tick + 1;
  if (state->baseline_count <= 1) {
    set_status(state, "There is no future interval to simulate.");
    state->phase = SHOT_FINDER_IDLE;
    return false;
  }

  const int global_start = state->earliest_fire_tick + state->group_start_offset;
  const ft_world *source = model_group_world_at_tick(timeline, state->group, global_start);
  if (!source) {
    set_status(state, "Could not obtain the starting world for the search.");
    state->phase = SHOT_FINDER_IDLE;
    return false;
  }

  state->player_count = gh_world_player_count(host, source);
  state->input_size = game_input_size(host);
  state->baseline = calloc((size_t)state->baseline_count, sizeof(*state->baseline));
  state->packed_inputs = calloc((size_t)state->player_count, state->input_size);
  if (state->kind == SHOT_FINDER_MULTI_EARLY) {
    state->planned_shot_capacity = state->latest_fire_tick - state->earliest_fire_tick + 1;
    state->planned_shots = calloc((size_t)state->planned_shot_capacity, sizeof(*state->planned_shots));
  }
  // These are prediction worlds, never presentation worlds. Index -1 is the
  // game ABI's explicit scratch identity and prevents speculative shots from
  // emitting particles/events into the visible simulation group.
  state->baseline_world = gh_world_create(host, ui->gfx_handler->level, state->player_count, -1);
  state->candidate_seed_world = gh_world_create(host, ui->gfx_handler->level, state->player_count, -1);
  state->candidate_world = gh_world_create(host, ui->gfx_handler->level, state->player_count, -1);
  if (!state->baseline || !state->packed_inputs || !state->baseline_world || !state->candidate_seed_world ||
      !state->candidate_world ||
      (state->kind == SHOT_FINDER_MULTI_EARLY && !state->planned_shots)) {
    set_status(state, "Not enough memory to start the search.");
    release_search(ui, false);
    return false;
  }

  gh_world_copy(host, state->baseline_world, source);
  state->baseline[0] = sample_player(host, state->baseline_world, state->player);
  state->baseline_tick = state->earliest_fire_tick;
  state->seed_tick = -1;
  state->coarse_frame = searches_past(state)
                            ? state->latest_fire_tick
                            : state->selected_local_tick;
  state->coarse_angle = 0;
  state->refine_angle = 0;
  state->completed_candidates = 0;
  const int frames = searches_past(state)
                         ? state->latest_fire_tick - state->earliest_fire_tick + 1
                         : 1;
  state->total_candidates = frames * SHOT_FINDER_COARSE_ANGLES + SHOT_FINDER_REFINE_ANGLES;
  state->have_best = false;
  state->best_score = -DBL_MAX;
  state->best_maximum_progress = -DBL_MAX;
  state->best_maximum_velocity = -DBL_MAX;
  state->best_impact_velocity = -DBL_MAX;
  state->best_target_velocity = -DBL_MAX;
  state->best_impact_tick = -1;
  state->planned_shot_count = 0;
  state->planned_maximum_progress = -DBL_MAX;
  state->planned_target_velocity = 0.0;
  state->candidate_active = false;
  state->phase = SHOT_FINDER_BUILD_BASELINE;
  set_status(state, "Building the authored future trajectory...");
  return true;
}

static void begin_search(ui_handler_t *ui, ft_vec2 direction) {
  shot_finder_t *state = finder(ui);
  const float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
  if (length < 0.001f) {
    set_status(state, "Choose a direction farther from the player.");
    return;
  }
  state->desired_direction = (ft_vec2){direction.x / length, direction.y / length};
  if (!resolve_search_schema(ui)) return;

  state->earliest_fire_tick = state->selected_local_tick;
  state->latest_fire_tick = state->selected_local_tick;
  if (state->kind == SHOT_FINDER_EARLY) {
    if (state->selected_local_tick < SHOT_FINDER_EARLY_MIN_LEAD) {
      set_status(state, "There is no earlier frame from which to send a grenade.");
      state->phase = SHOT_FINDER_IDLE;
      return;
    }
    state->scan_tick = state->selected_local_tick - SHOT_FINDER_EARLY_MIN_LEAD;
    state->scan_found_ready = false;
    state->latest_fire_tick = -1;
    state->phase = SHOT_FINDER_SCAN_EARLY;
    set_status(state, "Finding legal grenade frames before the target...");
  } else if (state->kind == SHOT_FINDER_MULTI_EARLY) {
    if (state->selected_local_tick < SHOT_FINDER_EARLY_MIN_LEAD) {
      set_status(state, "There is no earlier frame from which to send grenades.");
      state->phase = SHOT_FINDER_IDLE;
      return;
    }
    // Combination mode must be able to cross several cooldown windows. Test
    // the entire authored past and let DDNet's simulation reject illegal
    // firing ticks; processing remains incremental regardless of duration.
    state->earliest_fire_tick = 0;
    state->latest_fire_tick = state->selected_local_tick - SHOT_FINDER_EARLY_MIN_LEAD;
    allocate_search(ui);
  } else {
    allocate_search(ui);
  }
}

static bool pack_inputs(ui_handler_t *ui, int tick, const input_record_t *override) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;
  if (!state->packed_inputs || state->input_size == 0) return false;

  for (int player = 0; player < state->player_count; ++player) {
    const int track = model_group_track_index(timeline, state->group, player);
    input_record_t record;
    if (track >= 0)
      record = model_get_input_at_tick(timeline, track, tick);
    else
      engine_input_default(host, &record);
    if (override && track == state->track) record = *override;
    memcpy(state->packed_inputs + (size_t)player * state->input_size, record.bytes, state->input_size);
  }
  return true;
}

static void update_early_scan(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;
  const int global_tick = state->scan_tick + state->group_start_offset;
  const ft_world *world = model_group_world_at_tick(timeline, state->group, global_tick);
  int reload = 0;
  bool has_grenade = false;
  if (!world || !player_int(host, world, state->player, state->reload_timer_prop, &reload) ||
      !player_bool(host, world, state->player, state->has_grenade_prop, &has_grenade)) {
    set_status(state, "Could not inspect grenade ownership/cooldown at track tick %d.", state->scan_tick);
    state->phase = SHOT_FINDER_IDLE;
    return;
  }

  const bool ready = has_grenade && reload <= 0;
  if (!state->scan_found_ready) {
    // The selected target may already be inside the cooldown caused by an
    // authored shot. Walk through that cooldown to the frame on which a
    // grenade could actually have been fired, then search its ready window.
    if (ready) {
      state->scan_found_ready = true;
      state->latest_fire_tick = state->scan_tick;
      state->earliest_fire_tick = state->scan_tick;
    }
  } else if (!ready) {
    allocate_search(ui);
    return;
  } else {
    state->earliest_fire_tick = state->scan_tick;
  }

  if (state->scan_tick == 0) {
    if (state->scan_found_ready)
      allocate_search(ui);
    else {
      set_status(state, "No legal grenade firing frame was found before the selected target.");
      state->phase = SHOT_FINDER_IDLE;
    }
    return;
  }
  --state->scan_tick;
}

static void update_baseline(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  game_host_t *host = &ui->gfx_handler->game_host;
  if (state->baseline_tick >= state->evaluation_end_tick) {
    state->phase = SHOT_FINDER_SEARCH_COARSE;
    set_status(state, "Searching every two-degree aim angle...");
    return;
  }
  if (!pack_inputs(ui, state->baseline_tick, NULL)) {
    set_status(state, "Could not read authored TAS inputs.");
    release_search(ui, false);
    return;
  }
  gh_world_step(host, state->baseline_world, state->packed_inputs, (unsigned)state->player_count);
  state->baseline_tick = gh_world_tick(host, state->baseline_world);
  const int index = state->baseline_tick - state->earliest_fire_tick;
  if (index < 0 || index >= state->baseline_count) {
    set_status(state, "The game simulation returned an unexpected tick.");
    release_search(ui, false);
    return;
  }
  state->baseline[index] = sample_player(host, state->baseline_world, state->player);
}

static ft_vec2 aim_for_angle(double degrees) {
  const double radians = degrees * M_PI / 180.0;
  return (ft_vec2){(float)(cos(radians) * 1000.0), (float)(-sin(radians) * 1000.0)};
}

static bool find_candidate_grenade(shot_finder_t *state, game_host_t *host, const ft_world *world) {
  if (!state || !host || !world || state->projectile_class < 0) return false;
  const int count = gh_entity_count(host, world, (unsigned)state->projectile_class);
  int continuation_start_tick = -1;
  int continuation_lifespan = -1;
  for (int entity = 0; entity < count; ++entity) {
    int type = -1;
    int owner = -1;
    int start_tick = -1;
    int lifespan = -1;
    bool explosive = false;
    if (!entity_int(host, world, state->projectile_class, entity, state->projectile_type_prop, &type) ||
        !entity_int(host, world, state->projectile_class, entity, state->projectile_owner_prop, &owner) ||
        !entity_int(host, world, state->projectile_class, entity, state->projectile_start_tick_prop,
                    &start_tick) ||
        !entity_int(host, world, state->projectile_class, entity, state->projectile_lifespan_prop,
                    &lifespan) ||
        !entity_bool(host, world, state->projectile_class, entity, state->projectile_explosive_prop,
                     &explosive))
      continue;
    if (type != state->grenade_weapon || owner != state->player || !explosive)
      continue;
    if (state->candidate_projectile_start_tick < 0) {
      // Discover the start tick from the newly created entity itself. DDNet's
      // attack and projectile clocks normally agree, but the entity property
      // is the authoritative identity and avoids baking that assumption in.
      if (start_tick < state->candidate_fire_tick || start_tick > state->candidate_sim_tick) continue;
      state->candidate_projectile_start_tick = start_tick;
      state->candidate_projectile_lifespan = lifespan;
      return true;
    }
    if (start_tick == state->candidate_projectile_start_tick) {
      state->candidate_projectile_lifespan = lifespan;
      return true;
    }
    // DDNet may restart a projectile's ballistic start tick when it teleports
    // or changes trajectory. Its decremented lifespan still identifies the
    // same grenade without relying on an unstable entity-list index.
    if (lifespan == state->candidate_projectile_lifespan - 1) {
      continuation_start_tick = start_tick;
      continuation_lifespan = lifespan;
    }
  }
  if (continuation_start_tick >= 0) {
    state->candidate_projectile_start_tick = continuation_start_tick;
    state->candidate_projectile_lifespan = continuation_lifespan;
    return true;
  }
  return false;
}

static int planned_shot_index_at(const shot_finder_t *state, int tick) {
  if (!state) return -1;
  for (int i = 0; i < state->planned_shot_count; ++i)
    if (state->planned_shots[i].fire_tick == tick) return i;
  return -1;
}

static void configure_grenade_input(game_host_t *host, shot_finder_t *state,
                                    input_record_t *input, double angle) {
  engine_input_set_vec2(host, input, state->target_field, aim_for_angle(angle));
  engine_input_set(host, input, state->fire_field, 1);
  engine_input_set(host, input, state->weapon_field, state->grenade_weapon);
}

static bool prepare_candidate(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;

  int fire_tick;
  double angle;
  if (state->phase == SHOT_FINDER_SEARCH_COARSE) {
    if (state->coarse_frame < state->earliest_fire_tick) {
      if (!state->have_best) return false;
      state->phase = SHOT_FINDER_SEARCH_REFINE;
      state->refine_angle = 0;
      state->refine_frame = state->best_fire_tick;
      state->refine_center_angle = state->best_angle;
      state->candidate_active = false;
      set_status(state, "Refining the best frame and angle...");
      return true;
    }
    fire_tick = state->coarse_frame;
    angle = state->coarse_angle * (360.0 / SHOT_FINDER_COARSE_ANGLES);
  } else {
    if (state->refine_angle >= SHOT_FINDER_REFINE_ANGLES) return false;
    fire_tick = state->refine_frame;
    angle = state->refine_center_angle - 2.0 + state->refine_angle * 0.1;
  }

  int simulation_start_tick = fire_tick;
  if (state->kind == SHOT_FINDER_MULTI_EARLY) {
    for (int i = 0; i < state->planned_shot_count; ++i)
      if (state->planned_shots[i].fire_tick < simulation_start_tick)
        simulation_start_tick = state->planned_shots[i].fire_tick;
  }
  if (state->seed_tick != simulation_start_tick) {
    const int global_tick = simulation_start_tick + state->group_start_offset;
    const ft_world *source = model_group_world_at_tick(timeline, state->group, global_tick);
    if (!source) return false;
    gh_world_copy(host, state->candidate_seed_world, source);
    state->seed_tick = simulation_start_tick;
  }
  gh_world_copy(host, state->candidate_world, state->candidate_seed_world);
  state->candidate_input = model_get_input_at_tick(timeline, state->track, fire_tick);
  engine_input_set_vec2(host, &state->candidate_input, state->target_field, aim_for_angle(angle));
  engine_input_set(host, &state->candidate_input, state->fire_field, 1);
  if (searches_past(state))
    engine_input_set(host, &state->candidate_input, state->weapon_field, state->grenade_weapon);
  else
    engine_input_set(host, &state->candidate_input, state->weapon_field, state->selected_weapon);

  state->candidate_fire_tick = fire_tick;
  state->candidate_sim_tick = simulation_start_tick;
  state->candidate_angle = angle;
  state->candidate_score = (candidate_score_t){.maximum_progress = -DBL_MAX,
                                                .maximum_velocity = -DBL_MAX,
                                                .impact_velocity = -DBL_MAX,
                                                .target_velocity = -DBL_MAX,
                                                .impact_tick = -1};
  state->candidate_projectile_start_tick = -1;
  state->candidate_projectile_lifespan = -1;
  state->candidate_projectile_alive = false;
  state->candidate_active = true;
  return true;
}

static double score_candidate(const candidate_score_t *score) {
  if (!score || score->samples <= 0 || score->maximum_progress == -DBL_MAX) return -DBL_MAX;
  const double average_progress = score->weighted_progress / score->samples;
  const double average_lateral = score->lateral_distance / score->samples;
  return 0.55 * score->maximum_progress + 0.25 * average_progress + 0.20 * score->end_progress +
         8.0 * score->maximum_velocity - 0.04 * average_lateral - 10.0 * score->lost_alive_samples;
}

static void advance_candidate_index(shot_finder_t *state) {
  ++state->completed_candidates;
  if (state->phase == SHOT_FINDER_SEARCH_COARSE) {
    ++state->coarse_angle;
    if (state->coarse_angle >= SHOT_FINDER_COARSE_ANGLES) {
      state->coarse_angle = 0;
      if (searches_past(state))
        --state->coarse_frame;
      else
        state->coarse_frame = state->earliest_fire_tick - 1;
    }
  } else {
    ++state->refine_angle;
  }
}

static void finish_candidate(shot_finder_t *state, bool fired) {
  if (fired) {
    const double score = score_candidate(&state->candidate_score);
    const double average_progress = state->candidate_score.samples > 0
                                        ? state->candidate_score.weighted_progress / state->candidate_score.samples
                                        : -DBL_MAX;
    const bool useful_direction = state->candidate_score.maximum_progress > SHOT_FINDER_DIRECTION_EPSILON &&
                                  (state->candidate_score.end_progress > 0.0 || average_progress > 0.0);
    const bool timed_impact = !searches_past(state) ||
                              (state->candidate_score.impact_near_target &&
                               state->candidate_score.impact_velocity > SHOT_FINDER_SPEED_EPSILON);
    const double ranking_velocity = searches_past(state)
                                        ? state->candidate_score.target_velocity
                                        : state->candidate_score.maximum_velocity;
    const double best_ranking_velocity = searches_past(state)
                                             ? state->best_target_velocity
                                             : state->best_maximum_velocity;
    const bool prompt_effect = state->kind != SHOT_FINDER_CURRENT ||
                               state->candidate_score.effect_within_current_window;
    const bool improves_plan = state->kind != SHOT_FINDER_MULTI_EARLY ||
                               ranking_velocity > state->planned_target_velocity + SHOT_FINDER_SPEED_EPSILON;
    const bool acceptable = timed_impact && prompt_effect && improves_plan && useful_direction &&
                            ranking_velocity > SHOT_FINDER_SPEED_EPSILON &&
                            state->candidate_score.lost_alive_samples == 0 &&
                            score > SHOT_FINDER_IMPROVEMENT_EPSILON;
    const bool faster = !state->have_best || ranking_velocity > best_ranking_velocity + SHOT_FINDER_SPEED_EPSILON;
    const bool same_speed_better_movement = state->have_best &&
                                            fabs(ranking_velocity - best_ranking_velocity) <=
                                                SHOT_FINDER_SPEED_EPSILON &&
                                            score > state->best_score;
    // Past-shot modes rank the combined directional speed around the target.
    // Only their post-target movement score is allowed to break power ties.
    if (acceptable && (faster || same_speed_better_movement)) {
      state->have_best = true;
      state->best_score = score;
      state->best_maximum_progress = state->candidate_score.maximum_progress;
      state->best_maximum_velocity = state->candidate_score.maximum_velocity;
      state->best_impact_velocity = state->candidate_score.impact_velocity;
      state->best_target_velocity = state->candidate_score.target_velocity;
      state->best_impact_tick = state->candidate_score.impact_tick;
      state->best_angle = state->candidate_angle;
      state->best_fire_tick = state->candidate_fire_tick;
    }
  }
  state->candidate_active = false;
  advance_candidate_index(state);
}

static bool update_candidate(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;
  if (!state->candidate_active && !prepare_candidate(ui)) return false;
  // prepare_candidate may switch from the coarse pass into refinement without
  // starting a candidate; the next budgeted iteration will start it.
  if (!state->candidate_active) return true;
  if (state->candidate_sim_tick >= state->evaluation_end_tick) {
    finish_candidate(state, true);
    return true;
  }
  const int step_tick = state->candidate_sim_tick;
  const bool trial_shot_step = step_tick == state->candidate_fire_tick;
  bool inserted_shot_step = trial_shot_step;
  input_record_t combined_input;
  const input_record_t *override = trial_shot_step ? &state->candidate_input : NULL;
  if (state->kind == SHOT_FINDER_MULTI_EARLY) {
    const int planned_index = planned_shot_index_at(state, step_tick);
    inserted_shot_step = trial_shot_step || planned_index >= 0;
    if (inserted_shot_step) {
      combined_input = model_get_input_at_tick(timeline, state->track, step_tick);
      const double angle = trial_shot_step ? state->candidate_angle
                                           : state->planned_shots[planned_index].angle;
      configure_grenade_input(host, state, &combined_input, angle);
      override = &combined_input;
    } else {
      override = NULL;
    }
  }

  int attack_before = 0;
  int reload_before = 0;
  if (inserted_shot_step &&
      (!player_int(host, state->candidate_world, state->player, state->attack_tick_prop, &attack_before) ||
       !player_int(host, state->candidate_world, state->player, state->reload_timer_prop, &reload_before))) {
    finish_candidate(state, false);
    return true;
  }
  if (!pack_inputs(ui, step_tick, override)) {
    finish_candidate(state, false);
    return true;
  }
  gh_world_step(host, state->candidate_world, state->packed_inputs, (unsigned)state->player_count);
  state->candidate_sim_tick = gh_world_tick(host, state->candidate_world);

  if (inserted_shot_step) {
    int attack_tick = attack_before;
    int reload_timer = reload_before;
    int active_weapon = -1;
    const bool have_attack = player_int(host, state->candidate_world, state->player,
                                        state->attack_tick_prop, &attack_tick);
    const bool have_reload = player_int(host, state->candidate_world, state->player,
                                        state->reload_timer_prop, &reload_timer);
    const bool have_weapon = player_int(host, state->candidate_world, state->player,
                                        state->active_weapon_prop, &active_weapon);
    const int requested_weapon = searches_past(state) ? state->grenade_weapon : state->selected_weapon;
    const bool attack_changed = attack_tick != attack_before;
    // A tick-zero shot legitimately writes attack tick 0 over its initial 0;
    // the reload timer is the disambiguating state change only in that case.
    const bool tick_zero_shot = step_tick == 0 && attack_before == 0 && attack_tick == 0 &&
                                reload_timer > reload_before;
    if (!have_attack || !have_reload || !have_weapon || active_weapon != requested_weapon ||
        (!attack_changed && !tick_zero_shot)) {
      // The real game rejected the shot (cooldown, freeze, ammo, weapon switch,
      // etc.). Do not reproduce those rules here; the simulation is the judge.
      finish_candidate(state, false);
      return true;
    }
    if (searches_past(state) && trial_shot_step) {
      // Entity list indices are not stable when other projectiles disappear,
      // so discover and then rediscover this exact grenade by its published
      // owner/type/start-tick identity on every simulated frame.
      if (!find_candidate_grenade(state, host, state->candidate_world)) {
        finish_candidate(state, false);
        return true;
      }
      state->candidate_projectile_alive = true;
    }
  }

  if (searches_past(state) && !trial_shot_step && state->candidate_projectile_alive) {
    if (!find_candidate_grenade(state, host, state->candidate_world)) {
      // The tracked grenade has just been removed by the real DDNet
      // simulation. A target-player velocity delta on this step identifies an
      // explosion interaction; a wall/lifespan explosion elsewhere has none.
      state->candidate_projectile_alive = false;
      state->candidate_score.impact_tick = state->candidate_sim_tick;
      const int impact_index = state->candidate_sim_tick - state->earliest_fire_tick;
      if (impact_index >= 0 && impact_index < state->baseline_count) {
        const shot_sample_t baseline = state->baseline[impact_index];
        const shot_sample_t candidate = sample_player(host, state->candidate_world, state->player);
        if (baseline.valid && candidate.valid) {
          const double velocity =
              ((candidate.velocity.x - baseline.velocity.x) * state->desired_direction.x +
               (candidate.velocity.y - baseline.velocity.y) * state->desired_direction.y) /
              32.0;
          state->candidate_score.impact_velocity = velocity;
          state->candidate_score.impact_near_target =
              abs(state->candidate_sim_tick - state->selected_local_tick) <= SHOT_FINDER_TARGET_WINDOW;
        }
      }
      if (!state->candidate_score.impact_near_target ||
          state->candidate_score.impact_velocity <= SHOT_FINDER_SPEED_EPSILON) {
        finish_candidate(state, false);
        return true;
      }
    }
  }

  if (searches_past(state) && !state->candidate_score.impact_near_target &&
      state->candidate_sim_tick > state->selected_local_tick + SHOT_FINDER_TARGET_WINDOW) {
    // Later explosions cannot satisfy this target. Stop this candidate early
    // instead of simulating the rest of the (potentially forty-second) future.
    finish_candidate(state, false);
    return true;
  }

  if (searches_past(state) &&
      abs(state->candidate_sim_tick - state->selected_local_tick) <= SHOT_FINDER_TARGET_WINDOW) {
    const int target_index = state->candidate_sim_tick - state->earliest_fire_tick;
    if (target_index >= 0 && target_index < state->baseline_count) {
      const shot_sample_t baseline = state->baseline[target_index];
      const shot_sample_t candidate = sample_player(host, state->candidate_world, state->player);
      if (baseline.valid && candidate.valid) {
        const double target_velocity =
            ((candidate.velocity.x - baseline.velocity.x) * state->desired_direction.x +
             (candidate.velocity.y - baseline.velocity.y) * state->desired_direction.y) /
            32.0;
        if (target_velocity > state->candidate_score.target_velocity)
          state->candidate_score.target_velocity = target_velocity;
      }
    }
  }

  // The firing-frame-to-target interval exists only to let the grenade travel.
  // Never reward movement during it. Once a correctly timed impact is known,
  // compare only the authored future after the selected target frame.
  if (state->candidate_sim_tick > state->selected_local_tick) {
    const int index = state->candidate_sim_tick - state->earliest_fire_tick;
    if (index >= 0 && index < state->baseline_count) {
      const shot_sample_t baseline = state->baseline[index];
      const shot_sample_t candidate = sample_player(host, state->candidate_world, state->player);
      if (baseline.valid && candidate.valid) {
        const double dx = candidate.position.x - baseline.position.x;
        const double dy = candidate.position.y - baseline.position.y;
        const double projected = dx * state->desired_direction.x + dy * state->desired_direction.y;
        const double perpendicular = fabs(-dx * state->desired_direction.y + dy * state->desired_direction.x);
        // DDNet reports position in tiles but velocity in physics pixels per
        // tick. Convert velocity advantage to the same scale as displacement.
        const double units_per_tile = 32.0;
        const double velocity = ((candidate.velocity.x - baseline.velocity.x) * state->desired_direction.x +
                                 (candidate.velocity.y - baseline.velocity.y) * state->desired_direction.y) /
                                units_per_tile;
        if (state->kind == SHOT_FINDER_CURRENT &&
            state->candidate_sim_tick <= state->selected_local_tick + SHOT_FINDER_CURRENT_EFFECT_WINDOW &&
            (projected > SHOT_FINDER_DIRECTION_EPSILON || velocity > SHOT_FINDER_SPEED_EPSILON))
          state->candidate_score.effect_within_current_window = true;
        const double span = state->evaluation_end_tick - state->selected_local_tick;
        const double time = span > 0 ? (state->candidate_sim_tick - state->selected_local_tick) / span : 1.0;
        const double weight = 0.35 + 0.65 * time;
        state->candidate_score.weighted_progress += projected * weight;
        state->candidate_score.lateral_distance += perpendicular;
        if (projected > state->candidate_score.maximum_progress)
          state->candidate_score.maximum_progress = projected;
        if (velocity > state->candidate_score.maximum_velocity)
          state->candidate_score.maximum_velocity = velocity;
        state->candidate_score.end_progress = projected;
        ++state->candidate_score.samples;
        if ((baseline.flags & FT_PLAYER_ALIVE) && !(candidate.flags & FT_PLAYER_ALIVE))
          ++state->candidate_score.lost_alive_samples;
      }
    }
  }

  if (state->candidate_sim_tick >= state->evaluation_end_tick) finish_candidate(state, true);
  return true;
}

static bool continue_multi_search(shot_finder_t *state) {
  if (!state || state->kind != SHOT_FINDER_MULTI_EARLY || !state->have_best)
    return false;

  const int replacement = planned_shot_index_at(state, state->best_fire_tick);
  if (replacement < 0 && state->planned_shot_count >= state->planned_shot_capacity) return false;
  planned_shot_t *shot = replacement >= 0
                             ? &state->planned_shots[replacement]
                             : &state->planned_shots[state->planned_shot_count++];
  *shot = (planned_shot_t){.fire_tick = state->best_fire_tick,
                           .impact_tick = state->best_impact_tick,
                           .angle = state->best_angle,
                           .impact_velocity = state->best_impact_velocity};
  state->planned_maximum_progress = state->best_maximum_progress;
  state->planned_target_velocity = state->best_target_velocity;

  const int frames = state->latest_fire_tick - state->earliest_fire_tick + 1;
  state->coarse_frame = state->latest_fire_tick;
  state->coarse_angle = 0;
  state->refine_angle = 0;
  state->completed_candidates = 0;
  state->total_candidates = frames * SHOT_FINDER_COARSE_ANGLES + SHOT_FINDER_REFINE_ANGLES;
  state->have_best = false;
  state->best_score = -DBL_MAX;
  state->best_maximum_progress = -DBL_MAX;
  state->best_maximum_velocity = -DBL_MAX;
  state->best_impact_velocity = -DBL_MAX;
  state->best_target_velocity = -DBL_MAX;
  state->best_impact_tick = -1;
  state->seed_tick = -1;
  state->candidate_active = false;
  state->phase = SHOT_FINDER_SEARCH_COARSE;
  if (replacement >= 0)
    set_status(state, "Improved the %d-grenade combination; re-optimizing frames and angles...",
               state->planned_shot_count);
  else
    set_status(state, "Locked in %d timed grenade%s; searching for another improvement...",
               state->planned_shot_count, state->planned_shot_count == 1 ? "" : "s");
  return true;
}

static input_record_t authored_input_at(timeline_state_t *timeline, int track_index, int tick) {
  player_track_t *track = &timeline->player_tracks[track_index];
  for (int i = 0; i < track->snippet_count; ++i) {
    input_snippet_t *snippet = &track->snippets[i];
    if (snippet->is_active && tick >= snippet->start_tick && tick < snippet->end_tick)
      return snippet_window(snippet)[tick - snippet->start_tick];
  }
  // A gap inherits the effective held input. model_apply_input_to_main_buffer
  // will create exactly one authored tick for the new shot.
  return model_get_input_at_tick(timeline, track_index, tick);
}

static void apply_best_shot(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  timeline_state_t *timeline = &ui->timeline;
  game_host_t *host = &ui->gfx_handler->game_host;

  if (state->kind == SHOT_FINDER_MULTI_EARLY) {
    if (state->planned_shot_count <= 0) {
      set_status(state, "No cooldown-valid grenade combination reached the selected target frame. The TAS was left unchanged.");
      release_search(ui, false);
      return;
    }
    if (state->track < 0 || state->track >= timeline->player_track_count || timeline->recording) {
      set_status(state, "The timeline changed while searching; the result was not applied.");
      release_search(ui, false);
      return;
    }
    timeline_data_snapshot_t *before = commands_capture_timeline_data(timeline);
    if (!before) {
      set_status(state, "Could not create an undo snapshot; the result was not applied.");
      release_search(ui, false);
      return;
    }
    int first_tick = state->planned_shots[0].fire_tick;
    for (int i = 0; i < state->planned_shot_count; ++i) {
      const planned_shot_t shot = state->planned_shots[i];
      input_record_t input = authored_input_at(timeline, state->track, shot.fire_tick);
      configure_grenade_input(host, state, &input, shot.angle);
      model_apply_input_to_main_buffer(timeline, &timeline->player_tracks[state->track],
                                       shot.fire_tick, &input);
      if (shot.fire_tick < first_tick) first_tick = shot.fire_tick;
    }
    model_recalc_physics(timeline, first_tick);
    undo_command_t *command = commands_create_timeline_data_change(ui, before,
                                                                   "Apply ShotFinder Multi Shot");
    if (command) undo_manager_register_command(&ui->undo_manager, command);

    const int shot_count = state->planned_shot_count;
    const double speed_gain = state->planned_target_velocity;
    const double distance_gain = state->planned_maximum_progress;
    const int target_frame = state->selected_global_tick;
    release_search(ui, false);
    set_status(state,
               "Applied %d timed grenades around target frame %d; combined directional speed +%.3f tiles/tick, peak gain %.3f tiles.",
               shot_count, target_frame, speed_gain, distance_gain);
    return;
  }

  const int best_tick = state->best_fire_tick;
  const int best_global = best_tick + state->group_start_offset;
  double shown_angle = fmod(state->best_angle, 360.0);
  if (shown_angle < 0.0) shown_angle += 360.0;

  if (!state->have_best || state->best_score <= SHOT_FINDER_IMPROVEMENT_EPSILON ||
      state->best_maximum_progress <= SHOT_FINDER_DIRECTION_EPSILON) {
    if (state->kind == SHOT_FINDER_EARLY)
      set_status(state,
                 "No grenade was found that boosts the chosen direction near the selected target frame. The TAS was left unchanged.");
    else if (state->kind == SHOT_FINDER_CURRENT)
      set_status(state, "No useful shot affected the chosen direction within five ticks. The TAS was left unchanged.");
    else
      set_status(state, "No useful improvement was found. The TAS was left unchanged.");
    release_search(ui, false);
    return;
  }
  if (state->kind == SHOT_FINDER_EARLY &&
      best_tick > state->selected_local_tick - SHOT_FINDER_EARLY_MIN_LEAD) {
    set_status(state, "The Early Shot result was not actually earlier; the TAS was left unchanged.");
    release_search(ui, false);
    return;
  }
  if (state->track < 0 || state->track >= timeline->player_track_count || timeline->recording) {
    set_status(state, "The timeline changed while searching; the result was not applied.");
    release_search(ui, false);
    return;
  }

  input_record_t input = authored_input_at(timeline, state->track, best_tick);
  engine_input_set_vec2(host, &input, state->target_field, aim_for_angle(state->best_angle));
  engine_input_set(host, &input, state->fire_field, 1);
  if (searches_past(state))
    engine_input_set(host, &input, state->weapon_field, state->grenade_weapon);
  else
    engine_input_set(host, &input, state->weapon_field, state->selected_weapon);

  timeline_data_snapshot_t *before = commands_capture_timeline_data(timeline);
  if (!before) {
    set_status(state, "Could not create an undo snapshot; the result was not applied.");
    release_search(ui, false);
    return;
  }
  model_apply_input_to_main_buffer(timeline, &timeline->player_tracks[state->track], best_tick, &input);
  model_recalc_physics(timeline, best_tick);
  undo_command_t *command = commands_create_timeline_data_change(ui, before, "Apply ShotFinder Shot");
  if (command) undo_manager_register_command(&ui->undo_manager, command);

  const double gain = state->best_maximum_progress;
  const double speed_gain = searches_past(state)
                                ? state->best_target_velocity
                                : state->best_maximum_velocity;
  const int flight_ticks = state->best_impact_tick - best_tick;
  const int impact_global = state->best_impact_tick + state->group_start_offset;
  const int impact_offset = state->best_impact_tick - state->selected_local_tick;
  release_search(ui, false);
  if (state->kind == SHOT_FINDER_EARLY)
    set_status(state,
               "Applied grenade at frame %d (%d-tick flight); predicted impact frame %d (%+d from target %d), aim %.1f degrees, boost +%.3f tiles/tick.",
               best_global, flight_ticks, impact_global, impact_offset, state->selected_global_tick,
               shown_angle, speed_gain);
  else
    set_status(state, "Applied an improved shot at frame %d (track tick %d), aim %.1f degrees; peak gain %.3f tiles, speed +%.3f tiles/tick.",
               best_global, best_tick, shown_angle, gain, speed_gain);
}

void shot_finder_update(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  if (!state || !search_running(state)) return;
  if (!is_ddnet(ui) || !ui->gfx_handler->level || state->track < 0 ||
      state->track >= ui->timeline.player_track_count || state->group < 0 ||
      state->group >= ui->timeline.group_count ||
      model_track_group_index(&ui->timeline, state->track) != state->group ||
      model_group_local_track_index(&ui->timeline, state->track) != state->player ||
      (state->player_count > 0 && model_group_track_count(&ui->timeline, state->group) != state->player_count) ||
      ui->timeline.groups[state->group]->start_offset != state->group_start_offset) {
    shot_finder_reset(ui, "The level or selected track changed; search cancelled.");
    return;
  }

  // ImGui's clock advances once per rendered frame, so it cannot enforce an
  // in-frame work budget. GLFW's timer is queried live while the loop runs.
  const double deadline = glfwGetTime() + SHOT_FINDER_FRAME_BUDGET;
  do {
    switch (state->phase) {
    case SHOT_FINDER_SCAN_EARLY:
      update_early_scan(ui);
      break;
    case SHOT_FINDER_BUILD_BASELINE:
      update_baseline(ui);
      break;
    case SHOT_FINDER_SEARCH_COARSE:
    case SHOT_FINDER_SEARCH_REFINE:
      if (!update_candidate(ui)) {
        if (continue_multi_search(state)) break;
        apply_best_shot(ui);
        return;
      }
      break;
    default:
      return;
    }
  } while (search_running(state) && glfwGetTime() < deadline);
}

static const char *weapon_name(int weapon) {
  static const char *const names[] = {"Hammer", "Gun", "Shotgun", "Grenade", "Laser", "Ninja"};
  return weapon >= 0 && weapon < (int)(sizeof(names) / sizeof(names[0])) ? names[weapon] : "Unknown";
}

void shot_finder_render_window(ui_handler_t *ui) {
  shot_finder_t *state = finder(ui);
  if (!state || !state->show_window) return;

  if (!igBegin("ShotFinder", &state->show_window, ImGuiWindowFlags_NoFocusOnAppearing)) {
    igEnd();
    return;
  }

  igTextWrapped("Search DDNet's real future simulation for a shot that improves movement in a general direction.");
  igSeparator();

  const bool available = is_ddnet(ui) && ui->gfx_handler->level &&
                         ui->timeline.selected_player_track_index >= 0 && !ui->timeline.recording;
  const bool busy = state->phase != SHOT_FINDER_IDLE;
  if (!available || busy) igBeginDisabled(true);
  const float width = igGetContentRegionAvail().x;
  if (igButton("Find best shot from where you are", (ImVec2){width, 0.f}))
    capture_selection(ui, SHOT_FINDER_CURRENT);
  if (igButton("Early Shot", (ImVec2){width, 0.f}))
    capture_selection(ui, SHOT_FINDER_EARLY);
  if (igButton("Multi Early Shot", (ImVec2){width, 0.f}))
    capture_selection(ui, SHOT_FINDER_MULTI_EARLY);
  if (!available || busy) igEndDisabled();

  if (busy) {
    if (igButton("Cancel", (ImVec2){0.f, 0.f})) shot_finder_reset(ui, "Search cancelled.");
    igSameLine(0.f, 8.f);
    if (state->phase == SHOT_FINDER_PICKING)
      igTextDisabled("Waiting for a viewport click");
    else
      igTextDisabled("Working incrementally; the editor remains responsive");
  }

  igSeparator();
  if (!is_ddnet(ui))
    igTextDisabled("Available for DDNet projects.");
  else if (!ui->gfx_handler->level)
    igTextDisabled("Load a map to use ShotFinder.");
  else if (ui->timeline.selected_player_track_index < 0)
    igTextDisabled("Select a player track and frame.");
  else {
    igText("Selected frame: %d", ui->timeline.current_tick);
    if (state->phase != SHOT_FINDER_IDLE)
      igText("Search weapon: %s", searches_past(state) ? "Grenade" : weapon_name(state->selected_weapon));
  }

  if (state->phase == SHOT_FINDER_SCAN_EARLY) {
    igText("Cooldown scan: track tick %d", state->scan_tick);
  } else if (state->phase == SHOT_FINDER_BUILD_BASELINE && state->baseline_count > 1) {
    const float progress = (float)(state->baseline_tick - state->earliest_fire_tick) /
                           (float)(state->baseline_count - 1);
    igProgressBar(progress, (ImVec2){-1.f, 0.f}, "Building baseline");
  } else if ((state->phase == SHOT_FINDER_SEARCH_COARSE || state->phase == SHOT_FINDER_SEARCH_REFINE) &&
             state->total_candidates > 0) {
    const float progress = (float)state->completed_candidates / (float)state->total_candidates;
    igProgressBar(progress, (ImVec2){-1.f, 0.f}, "Testing shots");
  }

  if (state->kind == SHOT_FINDER_MULTI_EARLY && search_running(state)) {
    igText("Combination so far: %d grenade%s", state->planned_shot_count,
           state->planned_shot_count == 1 ? "" : "s");
    if (state->planned_shot_count > 0)
      igText("Combined target speed: +%.3f tiles/tick", state->planned_target_velocity);
    const int shown_shots = state->planned_shot_count < 5 ? state->planned_shot_count : 5;
    for (int i = 0; i < shown_shots; ++i) {
      const planned_shot_t shot = state->planned_shots[i];
      igTextDisabled("  #%d fire %d -> impact %d, boost +%.3f", i + 1,
                     shot.fire_tick + state->group_start_offset,
                     shot.impact_tick + state->group_start_offset, shot.impact_velocity);
    }
    if (state->planned_shot_count > shown_shots)
      igTextDisabled("  ...and %d more", state->planned_shot_count - shown_shots);
  }

  if (state->have_best && search_running(state)) {
    double angle = fmod(state->best_angle, 360.0);
    if (angle < 0.0) angle += 360.0;
    const int global_tick = state->best_fire_tick + state->group_start_offset;
    igText("Best frame: %d (track tick %d)", global_tick, state->best_fire_tick);
    if (searches_past(state)) {
      igText("Lead: %d ticks before selected frame", state->selected_local_tick - state->best_fire_tick);
      igText("Predicted impact: frame %d (%+d ticks from target)",
             state->best_impact_tick + state->group_start_offset,
             state->best_impact_tick - state->selected_local_tick);
      igText("Directional boost at impact: %.3f tiles/tick", state->best_impact_velocity);
      if (state->kind == SHOT_FINDER_MULTI_EARLY)
        igText("Combined directional speed near target: %.3f tiles/tick", state->best_target_velocity);
    }
    igText("Best aim: %.1f degrees", angle);
    igText("Peak directional gain: %.3f tiles", state->best_maximum_progress);
    if (state->kind != SHOT_FINDER_EARLY)
      igText("Peak forward speed gain: %.3f tiles/tick", state->best_maximum_velocity);
  }

  igSpacing();
  igTextWrapped("%s", state->status);
  igSeparator();
  igTextDisabled("Full 360-degree search at 2-degree steps, then 0.1-degree refinement.");
  if (state->kind == SHOT_FINDER_EARLY)
    igTextDisabled("Early Shot requires the grenade to impact within 3 ticks of the selected frame.");
  else if (state->kind == SHOT_FINDER_MULTI_EARLY)
    igTextDisabled("Multi Early Shot keeps adding legal grenades while combined target speed improves.");
  else
    igTextDisabled("Current-frame shots must affect movement within 5 ticks; longer movement is still evaluated.");
  igEnd();
}

bool shot_finder_viewport_overlay(ui_handler_t *ui, ImVec2 viewport_origin, ImVec2 viewport_size,
                                  bool hovered) {
  shot_finder_t *state = finder(ui);
  if (!state || state->phase != SHOT_FINDER_PICKING) return false;

  if (igIsKeyPressed_Bool(ImGuiKey_Escape, false) || igIsMouseClicked_Bool(ImGuiMouseButton_Right, false)) {
    shot_finder_reset(ui, "Direction selection cancelled.");
    return false;
  }

  ImGuiIO *io = igGetIO_Nil();
  float player_x = 0.f, player_y = 0.f;
  world_to_screen(ui->gfx_handler, state->selection_origin.x, state->selection_origin.y, &player_x, &player_y);
  const ImVec2 player_screen = {viewport_origin.x + player_x, viewport_origin.y + player_y};
  const ImVec2 mouse = io->MousePos;
  ImDrawList *draw = igGetWindowDrawList();
  ImDrawList_AddLine(draw, player_screen, mouse, IM_COL32(255, 205, 55, 245), 3.f);
  ImDrawList_AddCircleFilled(draw, player_screen, 5.f, IM_COL32(255, 225, 95, 255), 16);

  const bool inside = mouse.x >= viewport_origin.x && mouse.x <= viewport_origin.x + viewport_size.x &&
                      mouse.y >= viewport_origin.y && mouse.y <= viewport_origin.y + viewport_size.y;
  if (hovered && inside) igSetTooltip("Click to choose a general movement direction");
  if (!hovered || !inside || !igIsMouseClicked_Bool(ImGuiMouseButton_Left, false)) return false;

  float world_x = 0.f, world_y = 0.f;
  screen_to_world(ui->gfx_handler, mouse.x - viewport_origin.x, mouse.y - viewport_origin.y, &world_x, &world_y);
  begin_search(ui, (ft_vec2){world_x - state->selection_origin.x, world_y - state->selection_origin.y});
  return true;
}
