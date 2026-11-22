#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model-visible text the agent loop injects, and its version (locked as agent/<version>)."""

from __future__ import annotations

AGENT_SURFACE_VERSION = 1
SELF_JUDGED_PROMPTS = frozenset({"amp1"})
SELF_JUDGED = {
    "finish_episode": ("End the episode now, scored as it stands. Use it once you have verified "
                       "the task is done, or when it has become impossible."),
    "submit_report": ("Your required last action once the episode is over: your account of the "
                      "episode. diagnosis is required unless you are confident the task "
                      "succeeded."),
    "diagnosis_required": ("diagnosis is required unless outcome_belief is success. Explain why "
                           "it failed or may have failed."),
    "episode_over": "The episode is over.",
    "rh": "rh (the robot SDK)",
}

EPISODE_OVER = {
    "over": "The episode is over (end: {end}).",
    "ended": (" It ended by itself: RoboLab ends an episode when its success check holds "
              "or when the step budget runs out."),
    "last_obs": (" The last observation before the end: step {step} of {max_steps} "
                 "({steps_left} left), mode {mode}."),
    "tail": (" Nothing more can be done in it. Now call submit_report (view_image is "
             "still available to look at saved images)."),
}
EPISODE_OVER_BENCHMARK = {
    "ended": (" It ended by itself: The benchmark ends an episode when its success "
              "check holds or when the step budget runs out."),
}


def self_judged(prompt: str) -> bool:
    return prompt in SELF_JUDGED_PROMPTS


def episode_over_text(end, obs: dict, benchmark: bool = False, judged_by_self: bool = False,
                      vdm: bool = False) -> str:
    """The model's episode-over tool result; benchmark: not RoboLab's robot; judged_by_self: says
    neither why nor when; vdm: a model that sees no images (no view_image)."""
    if judged_by_self:
        return SELF_JUDGED["episode_over"] + (VDM_NOTES["over_tail"] if vdm else
                                              EPISODE_OVER["tail"])
    msg = EPISODE_OVER["over"].format(end=end)
    if end == "ended":
        msg += (EPISODE_OVER_BENCHMARK if benchmark else EPISODE_OVER)["ended"]
    if obs.get("step") is not None:
        msg += EPISODE_OVER["last_obs"].format(
            step=obs.get("step"), max_steps=obs.get("max_steps"),
            steps_left=obs.get("steps_left"), mode=obs.get("mode"))
    return msg + (VDM_NOTES["over_tail"] if vdm else EPISODE_OVER["tail"])

VDM_SURFACE_VERSION = 5     # 4: the reasoning-budget bullet (amp-split patch 0010); 5: keep_tags


def vdm_surface_version(keep_tags: bool = False) -> int:
    """The version an episode records: 5 only with --keep-tags (amp-split patch 0017: the
    keep_tags() block in the tools section and the kernel function; 0015's inline-tag bullet,
    also 5, ran only in episode K). Without it the coder's surface is version 4's, and says so."""
    return VDM_SURFACE_VERSION if keep_tags else 4

VDM_NOTES = {
    "diff": "[vision model] What changed between step {before} and step {after}, seen by {seen}:"
            "\n{text}",
    "look": "[vision model] What the last motion changed (step {before} to step {after}), seen by "
            "{seen}:\n{text}",
    "scene": "[vision model] The scene at step {after}, seen by {seen}:\n{text}",
    "views": "{cams}, and your views: {paths}",
    "camera_view": "the {cam} camera",
    "unavailable": "[vision model] No description this time: {error}",
    "camera_only": "[vision model] Only cameras are described in this episode: show(camera=<name>, "
                   "caption=...). {n} image(s) you show()ed were saved but not described.",
    "no_camera": "[vision model] No camera {cam!r} in this episode; its cameras: {cams}.",
    "refused": "vision vdm_strict: this episode's code gets no camera, depth or perception data; "
               "show(camera=<name>, caption=...) asks the vision model",
    "over_tail": " Nothing more can be done in it. Now call submit_report.",
    "report_only": "The episode is over: only submit_report is available now.",
    "saved": " The images saved in this episode, which key_images may name: {paths}.",
    "saved_more": " ({n} earlier ones not listed)",
    "saved_none": " No image was saved in this episode: key_images may be empty.",
}

SAVED_LISTED = 40


def saved_images_text(paths: list[str]) -> str:
    """vision vdm: the saved images, for the report (history trimming drops the results that
    printed their paths, and the report phase runs no code)."""
    if not paths:
        return VDM_NOTES["saved_none"]
    more = len(paths) - SAVED_LISTED
    listed = ", ".join(paths[-SAVED_LISTED:]) + (VDM_NOTES["saved_more"].format(n=more)
                                                if more > 0 else "")
    return VDM_NOTES["saved"].format(paths=listed)

VDM_PROMPTS = {
    "task": "The robot's task: {instruction}",
    "initial_system": (
        "You are a helpful assistant that describes the initial state of the environment with "
        "the goal of the task in mind. You should try to provide objective information and no "
        "assumptions. Do *NOT* write any code."),
    "initial_user": (
        "Describe the initial state of the environment with the goal of the task in mind. You "
        "should try to provide objective information and no assumptions. Do *NOT* write any "
        "code."),
    "view": "{cam} camera view:",
    # amp-split, vision mode vdm: a scene call after the start, for the agent's questions
    "now_system": (
        "You are a helpful assistant that describes the current state of the environment, seen "
        "by the robot's cameras, and answers the questions of the agent controlling the robot, "
        "with the goal of the task in mind. You should try to provide objective information and "
        "no assumptions. Do *NOT* write any code."),
    "now_user": (
        "Describe the current state of the environment with the goal of the task in mind, and "
        "answer the questions of the agent controlling the robot below. You should try to provide "
        "objective information and no assumptions. Do *NOT* write any code."),
    "diff_system": (
        "You are a helpful assistant that describes the difference between the current state of "
        "the environment and the previous state of the environment with the goal of the task in "
        "mind and whether the task has been completed. You should try to provide objective "
        "information and no assumptions. Do *NOT* write any code."),
    "diff_user": (
        "Describe the difference between the current state of the environment and the previous "
        "state of the environment with the goal of the task in mind and whether the task has "
        "been completed. You should try to provide objective information and no assumptions. "
        "Do *NOT* write any code.."),
    "previous": "Previous state ({cam} camera):",
    "current": "Current state ({cam} camera):",
    "agent_view": "The agent controlling the robot also looked at the current state through this "
                  "view ({path}){caption}:",
    "caption": ', captioned "{caption}"',
    "agent_camera": "The agent controlling the robot asks about the {cam} camera{caption}.",
    "focus": 'The agent controlling the robot asks: "{captions}". Pay particular attention to '
             "this in your description.",
    "qualitative": ("Describe positions, sizes and distances only in words: relative to the "
                    "gripper, the robot and the other objects (left of, behind, touching, about "
                    "as wide as the open gripper). Give no pixel coordinates, distances, angles, "
                    "sizes or other numbers, even if the agent asks for them."),
    # amp-split, vision mode vdm: the agent measures positions itself and draws perceived ids on
    # the views it shows, so the vision model names drawn labels but still gives no numbers.
    "qualitative_vdm": (
        "Describe positions, sizes and distances only in words: relative to the gripper, the "
        "robot and the other objects (left of, behind, touching, about as wide as the open "
        "gripper). Give no pixel coordinates, distances, angles or sizes: the agent measures "
        "those itself. Where labels are drawn on an image (such as obj3), name the labelled "
        "objects by those labels."),
}

_SEEN_BY = """\
## How you see in this episode

This section overrides everything above, the reference included, about seeing images. You see no
images yourself: a separate vision model looks at the cameras for you and answers in text, inside
the result of your `python` call.

- Your first `python` result carries its description of the scene at the start. After every call
  that moved the robot (a motion or a gripper action), it describes what changed in every camera
  between just before the motion and just after it.
"""
_JUDGE = """\
- Wherever the text above tells you to look at the cameras or at an image, or to judge from
  images, ask the vision model instead, and judge from its answers together with the numbers you
  have. Its descriptions can be vague or wrong: check them against the numbers, and ask about a
  second camera before you rely on one.
- If you must decide yourself that the task is done, ask the vision model about each part of the
  instruction, about at least two cameras, before you finish: ask it to describe where each
  object is relative to its target rather than to answer yes or no, and fix anything unclear while
  steps remain.
"""
VISION_SECTIONS = {
    "vdm_strict": _SEEN_BY + """\
- `show(camera=<name>, caption=None)` asks it about one of the robot's cameras, by the name the
  robot's description gives it; write the caption as your question, for example
  `show(camera="wrist", caption="is the cube between the fingers?")`. It sees the plain camera
  frames: nothing you draw reaches it. An image you pass as `show(x)` is saved but not described.
  `view_image` is not available.
- **It answers only in words**, never with pixel coordinates, distances, angles or other numbers,
  even if you ask. Ask for relations: left or right of, in front of, behind, above, touching, and
  how far relative to the gripper or to other objects.
- **Your code gets no camera images**: the calls for them (camera frames, and annotating or
  rendering a camera view) are refused with "not available in the sandbox: vision vdm_strict".
  The "Perception" section at the start of the reference says what the harness withholds from
  every agent alike (depth, perceived objects, the table height, camera calibration). Where the
  text above or the reference advises camera images, depth, point clouds, picked pixels,
  perceived objects, calibration or triangulation, that does not apply: the vision model's words
  are your only view of the scene. You have the robot's proprioception (joint positions, gripper
  command and measured opening, end-effector and TCP pose), its model and kinematics, whatever
  the collision checks report, the instruction, the budget and the vision model's text.
- You know an object's position only roughly: close in on it in small steps from the gripper's
  known pose (move, ask where it now is relative to the fingers, correct) before you descend, and
  descend in short steps, since you are not given the table's height.
- In your report, `key_images` may be empty: the camera views you asked about are not saved.
""" + _JUDGE,
    # amp-split: the text agent on a harness with full numeric perception.
    "vdm": """\
## How you see in this episode

This section overrides everything above, the reference included, about seeing images. You see no
images yourself: a separate vision model looks for you and answers in text, inside the result of
your `python` call. Your code keeps everything else the reference describes: camera frames,
depth, point clouds, camera calibration, the perceived objects with their ids, positions and
sizes, `annotate`, the 3D views and the collision checks.

- Your first `python` result carries its description of the scene at the start. After every call
  that moved the robot (a motion or a gripper action), it describes what changed in every camera
  between just before the motion and just after it.
- `show(x, caption=None)` sends an image your code made to the vision model: an annotated camera
  view, a crop, a 3D view, a render. `show(camera=<name>, caption=None)` sends it the plain current
  frame of one camera. Write the caption as your question, for example
  `show(ep.annotate(cam, objects=True), caption="which id is the banana, which the bowl?")` or
  `show(camera="wrist", caption="is the banana between the fingers?")`. At most 6 per call. The
  result lists the saved path of each image you showed; use those paths in your report.
  `view_image` is not available.
- **Numbers come from your code; identity and verification come from the vision model.** Take
  positions, sizes, heights and distances from the perceived objects, depth and point clouds,
  never from the vision model. Ask the vision model what things are and whether something
  happened: which perceived id is the object the instruction means, whether perception merged or
  split objects, whether a grasp held, whether an object is in or on its target.
- **It answers only in words**: no pixel coordinates, distances, angles or sizes, even if you ask.
  It does name the ids drawn on an image you show it, so draw them: `ep.annotate(cam,
  objects=True)` puts every perceived id on a camera view. Ask for relations: left or right of, in
  front of, behind, above, touching.
- Wherever the text above or the reference tells you to look at the cameras, an image or a 3D
  view, or to check ids against the images, show() that image with your question and judge from
  the answer together with your numbers. Its descriptions can be vague or wrong: when they
  disagree with the numbers, ask again about a second camera, and trust the measurements for
  where things are.
- If you must decide yourself that the task is done, ask the vision model about each part of the
  instruction, about at least two cameras (two `show(camera=...)`, or two annotated views), before
  you finish: ask it to describe where each object is relative to its target rather than to
  answer yes or no, check its answer against the perceived positions, and fix anything unclear
  while steps remain.
- Each answer of the vision model takes time from your wall-clock budget: show what helps you
  decide the next step, not every camera after every call.
- In your report, `key_images` names images you showed, by the paths the results listed (the
  episode-over message lists them again).
""",
}


# amp-split patch: the last bullet of the vdm section when the coder has a reasoning budget
# (--thinking-budget); vLLM forces the end of its reasoning at that many tokens.
VDM_BUDGET_TEXT = (
    "- Your thinking is slow next to your code: thinking past {n} tokens in one reply is ended "
    "for you, and you then act. Do arithmetic, geometry and checks in python, with the "
    "perception data, `ep.check` and `ep.preview`, and think only to decide the next call.\n")


# amp-split patch 0017 (--keep-tags): the kernel's keep_tags(kind, note) keeps one line past the
# removal of old reasoning. Its explanation goes into the prompt's tools section, right after the
# paragraph on what every tool result carries (KEEP_TAGS_ANCHOR), not into the vision section:
# the coder reads the tools section for how its code's results work. 0015's inline <keep> tags
# in replies, as a bullet at the end of the vdm section, were never used (episodes K: 0 notes in
# 170 replies). The same text all episode: the system prompt heads every cached prefix.
KEEP_TAGS_ANCHOR = "Read it before you plan the next move.\n"
KEEP_TAGS_TEXT = (
    "\n"
    "**Your older reasoning is removed from this conversation when it grows long.** What you "
    "keep with `keep_tags(kind, note)` (inside `python`) stays: the line is printed into the "
    "tool result, which is kept where the reasoning was. After a call that verified a fact, "
    "failed, or finished a subgoal, keep one line about it in your next cell, with its "
    "evidence:\n"
    "- `keep_tags(\"fact\", \"mug handle points +y (ep.objects() at step 120)\")`: something "
    "verified, and how.\n"
    "- `keep_tags(\"fail\", \"grasp from above slips off the mug rim (steps 140-180); approach "
    "from the side\")`: an approach that failed, and why.\n"
    "- `keep_tags(\"scene\", \"obj6 is the red mug, on its side (vision model, step 7)\")`: what "
    "the vision model said an object is, with its id.\n"
    "- `keep_tags(\"plan\", \"1 mug upright: done; 2 move to the plate: next\")`: a subgoal and "
    "its status.\n"
    "- `keep_tags(\"done\", \"red mug upright at step 210, ep.check passed\")`: a subgoal "
    "finished.\n"
    "One line each (400 characters at most), a few per episode; a note you never need later "
    "costs nothing.\n")


def keep_tags_prompt(text: str) -> str:
    """The system text with KEEP_TAGS_TEXT inserted after KEEP_TAGS_ANCHOR (amp-split patch
    0017); unchanged when it already holds the block. ValueError when the anchor is missing: the
    condition must not run on a prompt it cannot explain itself in."""
    if KEEP_TAGS_TEXT in text:
        return text
    head, sep, tail = text.partition(KEEP_TAGS_ANCHOR)
    if not sep:
        raise ValueError("the system text has no tool-result paragraph to put keep_tags after "
                         f"({KEEP_TAGS_ANCHOR.strip()!r})")
    return head + sep + KEEP_TAGS_TEXT + tail


def vision_section(prompt: str, vision: str, thinking_budget: int | None = None) -> str:
    """The section a vision harness configuration's system text ends with: none for image ones.
    For vdm with a reasoning budget it ends with VDM_BUDGET_TEXT (amp-split patch)."""
    if vision == "image":
        return ""
    section = VISION_SECTIONS[vision]
    if vision == "vdm" and thinking_budget:
        section += VDM_BUDGET_TEXT.format(n=int(thinking_budget))
    return section
