"""Rich part-composition description generator for PartNet shapes.

Replaces the template "something with a board, a bar, and 5 legs" style with
natural-language descriptions that sentence-transformers can embed more
effectively.  Each part name is mapped to a functional/structural phrase
that describes what the part *does* or *is*, producing sentences like:

    "a table with a flat tabletop surface, four legs for support, and a
     connecting bar for stability"

instead of:

    "a Table with a board, a bar, and 5 legs"

The mapping is deterministic and cached — no external API calls.

Usage:
    from experiments.rich_part_descriptions import generate_rich_description

    desc = generate_rich_description("Table", [1, 7, 8],
                                     {1: 200, 7: 1, 8: 400},
                                     part_point_xyz=...)
    # → "a table with a flat tabletop surface, a sturdy connecting bar,
    #    and four legs providing structural support"
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the connected-component counter and PartNet name lookup from exp13
from experiments.exp13_part_text_retrieval import (
    PARTNET_PART_NAMES,
    COUNTABLE_PARTS,
    AN_PARTS,
    _count_instances_cc,
    get_partnet_part_name,
    _article,
    _pluralise,
)


# ── Part-name → functional phrase mapping ────────────────────────────────────
# Maps (category, raw_part_name) → functional description phrase.
# Category is included because the same part name (e.g. "body") has different
# functional meanings in different categories (Bottle vs Mug vs TrashCan).
#
# The phrase describes what the part *does* or *is* in natural language.
# For countable parts, {count} is replaced with the word number (one, two, etc.)
# or the digit for counts > 10.

# Format: each entry is a phrase template.  If the template contains {count},
# it's treated as a countable part and the count is substituted.
# If no {count}, the phrase is used as-is.

RICH_PART_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "Bag": {
        "bag body":        "a main bag body for carrying items",
        "handle":          "{count} handles for gripping",
        "shoulder strap":  "{count} shoulder strap for carrying on the shoulder",
    },
    "Bed": {
        "pillow":                          "{count} pillows for resting the head",
        "mattress":                         "a mattress for sleeping on",
        "frame horizontal hard surface":    "a horizontal frame surface for structural support",
        "frame horizontal surface bar":     "{count} horizontal frame bars for structural support",
        "bar stretcher":                    "{count} stretcher bars connecting the frame",
        "leg":                              "{count} legs supporting the bed frame",
        "surface base":                     "a base surface for the mattress",
        "bed side surface horizontal bar":   "{count} horizontal side bars on the bed frame",
        "bed side surface panel":           "{count} side panels on the bed frame",
        "bed side surface vertical bar":    "{count} vertical side bars on the bed frame",
        "bed post":                         "{count} vertical bed posts for structural support",
        "headboard":                        "a headboard at the head of the bed",
        "ladder vertical bar":              "{count} vertical bars forming a ladder",
        "rung":                             "{count} ladder rungs for climbing",
    },
    "Bottle": {
        "body":     "a cylindrical body holding liquid",
        "closure":  "a closure sealing the bottle top",
        "lid":      "a lid covering the bottle opening",
        "handle":   "{count} handle for holding the bottle",
        "neck":     "a narrow neck connecting the body to the opening",
        "mouth":    "a mouth opening for pouring",
    },
    "Bowl": {
        "container":          "a concave container for holding food or liquid",
        "containing things":  "contents held within the bowl",
        "bottom":             "a flat bottom resting on a surface",
    },
    "Chair": {
        "headrest":                    "a headrest for supporting the head",
        "head connector":               "{count} connectors linking the headrest to the back",
        "back surface vertical bar":   "{count} vertical back bars for back support",
        "back surface horizontal bar":  "{count} horizontal back bars for back support",
        "back single surface":          "a single back surface for leaning against",
        "back connector":               "{count} back connectors for structural support",
        "back support":                 "a back support for leaning against",
        "back frame vertical bar":      "{count} vertical frame bars on the back",
        "back frame horizontal bar":    "{count} horizontal frame bars on the back",
        "back holistic frame":          "a holistic back frame for structural support",
        "arm sofa style":               "{count} sofa-style armrests for resting arms",
        "arm horizontal bar":            "{count} horizontal armrest bars for resting arms",
        "arm near vertical bar":        "{count} vertical armrest bars for structural support",
        "arm writing table":             "a writing-table armrest surface",
        "arm holistic frame":            "{count} holistic armrest frames",
        "arm connector":                 "{count} arm connectors linking the armrest to the seat",
        "wheel":                         "{count} wheels for mobility",
        "caster stem":                   "{count} caster stems holding wheels",
        "leg":                           "{count} legs supporting the seat",
        "central support":               "a central support structure",
        "knob":                          "{count} adjustment knobs",
        "lever":                         "{count} control levers",
        "foot":                          "{count} feet at the base of the legs",
        "runner":                        "{count} runners for a rocking base",
        "bar stretcher":                 "{count} stretcher bars connecting the legs",
        "rocker":                        "{count} curved rockers for a rocking motion",
        "pedestal":                      "a pedestal base for central support",
        "seat support":                  "a seat support structure",
        "seat holistic frame":            "a holistic seat frame",
        "seat frame bar":                 "{count} seat frame bars for structural support",
        "seat surface bar":               "{count} seat surface bars forming the seating area",
        "seat single surface":            "a single seat surface for sitting on",
        "chair base":                     "a chair base for stability",
    },
    "Clock": {
        "frame":             "a frame enclosing the clock face",
        "surface":            "a clock face surface for displaying time",
        "foot":               "{count} feet for standing on a surface",
        "base surface":       "a base surface for stability",
        "box":                "a clock box housing the mechanism",
        "pendulum body":      "a pendulum body for timekeeping",
        "chain":              "{count} chains for the pendulum mechanism",
        "pendulum clock top": "a decorative top piece on the clock",
    },
    "Dishwasher": {
        "frame":       "a frame housing the dishwasher interior",
        "door frame":   "a door frame for the washing compartment",
        "handle":      "{count} handle for opening the door",
        "foot":        "{count} feet for leveling the appliance",
        "surface":     "interior surfaces for holding dishes",
    },
    "Display": {
        "screen frame":   "a frame surrounding the display screen",
        "screen":          "a screen for displaying content",
        "base support":    "a base support for the display",
        "surface":         "a surface on the display body",
    },
    "Door": {
        "outside frame":  "an outer frame enclosing the door",
        "fixed part":     "a fixed part of the door structure",
        "movable part":   "a movable part that swings open",
        "surface board":  "{count} surface boards forming the door panel",
    },
    "Earphone": {
        "earbud connector":      "{count} earbud connectors",
        "earbud frame":          "{count} earbud frames housing the speakers",
        "earbud pad":            "{count} earbud pads for comfort",
        "earbud connector wire": "{count} wires connecting the earbuds",
        "top band":              "a top band resting on the head",
        "earcup pad":            "{count} earcup pads for comfort",
        "earcup frame":          "{count} earcup frames housing the speakers",
        "earcup connector":      "{count} connectors linking the earcups to the band",
        "connector wire":        "{count} connector wires",
    },
    "Faucet": {
        "switch":              "{count} switches for controlling water flow",
        "hose":                "a hose for water supply",
        "tube":                "{count} tubes directing water flow",
        "horizontal support":  "{count} horizontal supports for structural stability",
        "vertical support":    "{count} vertical supports for structural stability",
        "surface base":        "a base surface for mounting",
    },
    "Hat": {
        "crown":  "a crown covering the top of the head",
        "brim":   "a brim extending from the base",
        "button": "a decorative button on top",
        "bill":   "a bill extending forward",
        "panel":  "{count} panels forming the hat structure",
    },
    "Keyboard": {
        "frame": "a frame enclosing the keys",
        "key":   "{count} keys for typing input",
    },
    "Knife": {
        "guard":   "a guard protecting the hand from the blade",
        "handle":  "a handle for gripping",
        "butt":    "a butt cap at the end of the handle",
        "blade":   "a blade for cutting",
        "bolster": "a bolster reinforcing the handle-blade junction",
    },
    "Lamp": {
        "lamp arm":                "{count} arms positioning the light",
        "lamp shade":              "a shade diffusing the light",
        "light bulb":              "{count} light bulbs for illumination",
        "lamp body":               "a body housing the lamp mechanism",
        "chain":                   "{count} chains for hanging or adjusting",
        "lamp base part":          "a base for stability and support",
        "cord":                    "a power cord",
        "lamp pole":               "a vertical pole for height",
        "lamp body solid":         "a solid body housing the lamp",
        "lamp arm straight bar":   "{count} straight arm bars positioning the light",
        "lamp arm curved bar":     "{count} curved arm bars positioning the light",
        "lamp finial":             "a decorative finial on top",
        "lamp wireframe fitter":   "a wireframe fitter holding the shade",
        "leg":                     "{count} legs supporting the lamp",
        "lamp post":               "a post for elevating the light",
        "street lamp base":        "a heavy base for a street lamp",
        "lamp cover frame top":     "a top cover frame",
        "lamp cover frame bottom":  "a bottom cover frame",
        "lamp cover frame bar":     "{count} cover frame bars",
        "lamp cover holder":        "a holder for the lamp cover",
        "connector":               "{count} connectors for structural support",
    },
    "Laptop": {
        "screen side": "a screen side displaying content",
        "base side":   "a base side housing the keyboard",
    },
    "Microwave": {
        "frame":      "a frame housing the microwave interior",
        "door frame": "a door frame for the cooking compartment",
        "handle":     "{count} handle for opening the door",
        "tray":       "a tray for placing food inside",
        "foot":       "{count} feet for stability",
    },
    "Mug": {
        "body":             "a cylindrical body for holding liquid",
        "handle":           "{count} handle for gripping",
        "containing things": "contents held within the mug",
    },
    "Refrigerator": {
        "frame":      "a frame housing the refrigerator interior",
        "door frame": "{count} door frames for the compartments",
        "handle":     "{count} handles for opening doors",
        "shelf":      "{count} shelves for storing food",
        "foot":       "{count} feet for leveling the appliance",
        "surface":    "interior surfaces for food storage",
    },
    "Scissors": {
        "blade":  "{count} blades for cutting",
        "handle": "{count} handles for gripping",
    },
    "StorageFurniture": {
        "countertop":              "a countertop surface on top",
        "shelf":                   "{count} shelves for storage",
        "frame vertical bar":      "{count} vertical frame bars for structural support",
        "back panel":              "a back panel for structural integrity",
        "top panel":               "a top panel surface",
        "vertical side panel":     "{count} vertical side panels",
        "frame horizontal bar":    "{count} horizontal frame bars for structural support",
        "vertical front panel":    "{count} vertical front panels",
        "bottom panel":            "a bottom panel for structural support",
        "vertical divider panel":  "{count} vertical divider panels separating compartments",
        "drawer back":             "{count} drawer back panels",
        "drawer bottom":           "{count} drawer bottom panels",
        "drawer side":             "{count} drawer side panels",
        "drawer front":            "{count} drawer front panels",
        "handle":                  "{count} handles for opening drawers or doors",
        "base side panel":         "{count} base side panels for structural support",
        "foot":                    "{count} feet for stability",
        "wheel":                   "{count} wheels for mobility",
        "caster stem":             "{count} caster stems holding wheels",
        "hinge":                   "{count} hinges for door movement",
        "cabinet door surface":    "{count} cabinet door surfaces",
    },
    "Table": {
        "ping pong net":        "a net for table tennis",
        "tabletop surface":    "a flat tabletop surface for placing items",
        "bar stretcher":       "{count} stretcher bars connecting the legs",
        "leg":                 "{count} legs supporting the tabletop",
        "pool ball":           "{count} pool balls",
        "bar":                 "{count} connecting bars for structural support",
        "board":               "a board forming the tabletop",
        "tabletop dropleaf":   "{count} dropleaf extensions for the tabletop",
        "bench":               "a bench for seating",
        "bench connector":     "{count} connectors linking the bench",
        "glass":               "a glass tabletop surface",
        "central support":     "a central support for the tabletop",
        "tabletop connector":  "{count} connectors joining tabletop sections",
        "runner":              "{count} runners for a sliding mechanism",
        "circular stretcher":  "a circular stretcher connecting the legs",
        "foot":                "{count} feet at the base of the legs",
        "wheel":               "{count} wheels for mobility",
        "caster stem":         "{count} caster stems holding wheels",
        "back panel":          "a back panel for structural support",
        "vertical side panel": "{count} vertical side panels",
        "shelf":               "{count} shelves for storage",
        "handle":              "{count} handles for opening drawers",
        "cabinet door surface": "{count} cabinet door surfaces",
        "drawer back":          "{count} drawer back panels",
        "drawer bottom":       "{count} drawer bottom panels",
        "drawer side":          "{count} drawer side panels",
        "drawer front":        "{count} drawer front panels",
        "keyboard tray surface": "a keyboard tray surface",
        "vertical front panel": "{count} vertical front panels",
        "vertical divider panel": "{count} vertical divider panels",
        "pedestal":            "a pedestal base for central support",
    },
    "TrashCan": {
        "container bottom":  "a bottom surface of the container",
        "container box":     "a container body for holding trash",
        "container neck":   "a neck constraining the opening",
        "frame vertical bar":   "{count} vertical frame bars for structural support",
        "frame horizontal circle": "{count} horizontal frame circles for structural support",
        "frame bottom":     "a frame bottom for stability",
        "frame holistic":   "a holistic frame structure",
        "foot":             "{count} feet for stability",
        "cover support":    "a support for the lid cover",
        "cover lid":        "a lid covering the trash can",
    },
    "Vase": {
        "lid":    "a lid covering the vase opening",
        "container": "a container body for holding flowers or liquid",
        "foot":   "{count} feet for stability",
        "plant":  "a plant held within the vase",
        "liquid or soil": "liquid or soil contents",
    },
}


# ── Number word mapping ──────────────────────────────────────────────────────
# For natural-sounding descriptions: "four legs" rather than "4 legs"

_NUMBER_WORDS: Dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}


def _number_word(n: int) -> str:
    """Convert a count to a natural number word, or digit string for large counts."""
    return _NUMBER_WORDS.get(n, str(n))


# Irregular singularisations: plural noun → singular noun
_SINGULARISE_MAP: Dict[str, str] = {
    "shelves":    "shelf",
    "leaves":     "leaf",
    "knives":     "knife",
    "halves":     "half",
    "selves":     "self",
    "wolves":     "wolf",
    "feet":       "foot",
    "geese":      "goose",
    "teeth":      "tooth",
    "men":        "man",
    "women":      "woman",
    "children":   "child",
    "rungs":      "rung",
    "feet":       "foot",
}


def _singularise_noun(word: str) -> str:
    """Singularise a plural noun word.

    Handles common cases: shelves→shelf, panels→panel, handles→handle.
    Falls back to stripping 's' for regular plurals.
    """
    lower = word.lower()
    if lower in _SINGULARISE_MAP:
        return _SINGULARISE_MAP[lower]
    if lower.endswith("ies"):
        return lower[:-3] + "y"
    if lower.endswith("es") and lower[-3] in "sxz":
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return lower[:-1]
    return lower


# ── Rich description generation ──────────────────────────────────────────────

def _deduplicate_part_names(
    part_ids_present: list[int],
    category: str,
) -> list[Tuple[int, str]]:
    """Return (part_id, part_name) pairs, deduplicated by part name.

    PartNet has duplicate part names at different IDs (e.g. Table has
    multiple "leg" entries at IDs 4, 8, 10, 18, 22, 31, 50). We keep only
    the first occurrence of each name.
    """
    seen = set()
    result = []
    for pid in part_ids_present:
        pname = get_partnet_part_name(category, pid)
        if pname in ("other", ""):
            continue
        if pname in seen:
            continue
        seen.add(pname)
        result.append((pid, pname))
    return result


def _get_rich_phrase(category: str, part_name: str, count: int) -> str:
    """Look up the rich functional phrase for a (category, part_name) pair.

    Falls back to a generic phrase if no mapping exists.
    Substitutes {count} with a natural number word.

    Templates with {count} use plural nouns (e.g. "{count} handles for gripping").
    For count=1, the plural noun is singularised and prefixed with an article.
    """
    cat_key = next(
        (k for k in RICH_PART_DESCRIPTIONS if k.lower() == category.lower()),
        None,
    )
    phrase = None
    if cat_key:
        phrase = RICH_PART_DESCRIPTIONS[cat_key].get(part_name)

    if phrase is None:
        # Generic fallback
        article = _article(part_name)
        if count == 1:
            return f"{article} {part_name}"
        else:
            return f"{_number_word(count)} {_pluralise(part_name, count)}"

    # Substitute count
    if "{count}" in phrase:
        if count == 1:
            # Template is like "{count} connecting bars for structural support"
            # → find and singularise the first *actual* plural noun, then add article.
            # We can't just singularise words[0] because multi-word phrases like
            # "connecting bars" or "vertical side panels" have adjectives first.
            # Strategy: scan left-to-right and singularise the first word that
            # actually changes under _singularise_noun (i.e. is a real plural).
            rest = phrase.replace("{count} ", "", 1)
            words = rest.split()
            singularised = False
            for i, word in enumerate(words):
                # Stop scanning at prepositions/conjunctions — noun is before these
                if word.lower() in ("for", "of", "at", "on", "in", "with",
                                    "and", "or", "to", "from", "by", "the"):
                    break
                singular = _singularise_noun(word)
                if singular != word.lower():
                    words[i] = singular
                    singularised = True
                    break
            # Fallback: if nothing changed, singularise the last word before a prep
            if not singularised and words:
                words[-1] = _singularise_noun(words[-1])
            # Derive the article from the first letter of the actual phrase,
            # not from the part name — e.g. "arm positioning..." needs "an".
            first_char = words[0][0].lower() if words else ""
            article = "an" if first_char in "aeiou" else "a"
            return f"{article} {' '.join(words)}"
        else:
            return phrase.replace("{count}", _number_word(count))

    # Non-countable phrase — use as-is
    return phrase


def generate_rich_description(
    class_name: str,
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
) -> str:
    """Generate a natural-language part-composition description for one PartNet shape.

    Uses functional/structural phrases instead of raw part names:
        "a table with a flat tabletop surface, four legs supporting the
         tabletop, and a connecting bar for structural support"

    Parameters
    ----------
    class_name         : e.g. "Table"
    part_ids_present   : sorted list of local part IDs present in this shape
    part_point_counts  : local_part_id → number of points with that label
    part_point_xyz     : local_part_id → [M, 3] XYZ coords of those points
                         If provided, uses connected components to count instances.

    Returns
    -------
    Natural-language description string.
    """
    if not part_ids_present:
        article = "an" if class_name[0].lower() in "aeiou" else "a"
        return f"{article} {class_name.lower()}"

    # Deduplicate part names and get (id, name) pairs
    part_pairs = _deduplicate_part_names(part_ids_present, class_name)

    if not part_pairs:
        article = "an" if class_name[0].lower() in "aeiou" else "a"
        return f"{article} {class_name.lower()}"

    # Build rich phrases with counts
    phrases = []
    for pid, pname in part_pairs:
        count = 1
        if pname in COUNTABLE_PARTS:
            max_count = COUNTABLE_PARTS[pname]
            if part_point_xyz is not None and pid in part_point_xyz:
                count = _count_instances_cc(part_point_xyz[pid], max_count)
            else:
                count = 1

        rich_phrase = _get_rich_phrase(class_name, pname, count)
        phrases.append(rich_phrase)

    # Join phrases into a natural sentence
    if len(phrases) == 1:
        parts_str = phrases[0]
    elif len(phrases) == 2:
        parts_str = f"{phrases[0]} and {phrases[1]}"
    else:
        parts_str = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    article = "an" if class_name[0].lower() in "aeiou" else "a"
    return f"{article} {class_name.lower()} with {parts_str}"


def generate_rich_agnostic_description(
    class_name: str,
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
    prefix: str = "a 3D model with",
) -> str:
    """Generate a class-agnostic rich description.

    Strips the class name and uses a generic prefix instead:
        "a 3D model with a flat tabletop surface, four legs, and a connecting bar"

    This matches the evaluation query phrasing ("something with...") while
    keeping the rich functional part descriptions.
    """
    if not part_ids_present:
        return f"{prefix} parts"

    part_pairs = _deduplicate_part_names(part_ids_present, class_name)

    if not part_pairs:
        return f"{prefix} parts"

    phrases = []
    for pid, pname in part_pairs:
        count = 1
        if pname in COUNTABLE_PARTS:
            max_count = COUNTABLE_PARTS[pname]
            if part_point_xyz is not None and pid in part_point_xyz:
                count = _count_instances_cc(part_point_xyz[pid], max_count)

        rich_phrase = _get_rich_phrase(class_name, pname, count)
        phrases.append(rich_phrase)

    if len(phrases) == 1:
        parts_str = phrases[0]
    elif len(phrases) == 2:
        parts_str = f"{phrases[0]} and {phrases[1]}"
    else:
        parts_str = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    return f"{prefix} {parts_str}"


def generate_rich_agnostic_descriptions_all(
    class_name: str,
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
    prefixes: list[str] = None,
) -> list[str]:
    """Generate ALL class-agnostic rich description variants.

    Returns one description per prefix. Default uses a single prefix
    for efficiency (matching the --n_agnostic=1 setting).
    """
    if prefixes is None:
        prefixes = ["a 3D model with"]

    return [
        generate_rich_agnostic_description(
            class_name, part_ids_present, part_point_counts,
            part_point_xyz, prefix=prefix,
        )
        for prefix in prefixes
    ]


def build_rich_shape_descriptions_dual(
    dataset,
    categories: dict,
    dataset_name: str = "partnet",
    agnostic_prefixes: list[str] = None,
) -> tuple[list[str], list[list[str]]]:
    """Generate rich class-bearing and class-agnostic descriptions for all shapes.

    Works with both PartNet (per-category 0-based part IDs, 24 classes) and
    ShapeNet (global 50 part IDs, 16 classes). For ShapeNet, falls back to
    the original template descriptions from exp13.

    Returns (class_descs, agnostic_descs_all) — two lists aligned with
    dataset indices.

    For PartNet, each shape automatically gets one extra class-hybrid agnostic
    view appended beyond the static agnostic_prefixes.  The hybrid prefix is:
        "a {class_name}, specifically a 3D model with ..."
    This means with the default agnostic_prefixes=["a 3D model with"], the
    cache will contain 2 agnostic views per shape (use --n_agnostic=2 to see
    both during training).

    Parameters
    ----------
    dataset            : ShapeNet/PartNet dataset or Subset
    categories         : {int: str} category mapping
    dataset_name       : "partnet" or "shapenet"
    agnostic_prefixes  : list of static prefix strings for agnostic variants.
                         Default: ["a 3D model with"] (single generic view).
                         One class-hybrid view is always appended for PartNet.
    """
    from torch.utils.data import Subset

    if agnostic_prefixes is None:
        agnostic_prefixes = ["a 3D model with"]

    if isinstance(dataset, Subset):
        raw_ds  = dataset.dataset
        indices = dataset.indices
    else:
        raw_ds  = dataset
        indices = range(len(raw_ds))

    part_labels_all = raw_ds.part_labels
    labels_all      = raw_ds.labels
    points_all      = getattr(raw_ds, "points", None)
    is_partnet      = dataset_name.lower() == "partnet"

    class_descs        = []
    agnostic_descs_all = []

    for raw_i in indices:
        cls_idx   = int(labels_all[raw_i])
        cls_name  = categories.get(cls_idx, f"class{cls_idx}")
        per_point = part_labels_all[raw_i]

        cnt = Counter(per_point.tolist())

        # Get XYZ for connected-component counting
        part_xyz = None
        if points_all is not None:
            xyz = points_all[raw_i]
            if hasattr(xyz, "numpy"):
                xyz = xyz.numpy()
            if xyz.shape[0] == 3 and xyz.ndim == 2:
                xyz = xyz.T
            elif xyz.ndim == 3:
                xyz = xyz  # already [N, 3]

        if is_partnet:
            present = sorted(pid for pid in cnt if pid > 0)
            if part_xyz is not None and xyz is not None:
                part_xyz = {pid: xyz[per_point == pid] for pid in present}

            class_descs.append(
                generate_rich_description(cls_name, present, cnt, part_xyz)
            )

            # Build agnostic views: static prefixes + one class-hybrid prefix
            # e.g. "a chair, specifically a 3D model with four legs..."
            # This gives the model a view that bridges the class name with the
            # part-level grounding, improving zero-shot and part-query alignment.
            article = "an" if cls_name[0].lower() in "aeiou" else "a"
            hybrid_prefix = f"{article} {cls_name.lower()}, specifically a 3D model with"
            all_prefixes = list(agnostic_prefixes) + [hybrid_prefix]

            agnostic_descs_all.append(
                generate_rich_agnostic_descriptions_all(
                    cls_name, present, cnt, part_xyz,
                    prefixes=all_prefixes,
                )
            )
        else:
            # ShapeNet: fall back to original exp13 generators
            from experiments.exp13_part_text_retrieval import (
                generate_description,
                generate_agnostic_descriptions_all,
            )
            valid_ids = set()  # Will be filled from SHAPE_NET_PART_SEG_CLASSES
            from experiments.exp13_part_text_retrieval import SHAPE_NET_PART_SEG_CLASSES
            valid_ids = set(SHAPE_NET_PART_SEG_CLASSES[cls_idx])
            present = sorted(pid for pid in cnt if pid in valid_ids)

            if part_xyz is not None and xyz is not None:
                part_xyz = {pid: xyz[per_point == pid] for pid in present}

            # Temporarily override PART_ID_TO_INFO for this shape
            from experiments.exp13_part_text_retrieval import PART_ID_TO_INFO
            _orig = dict(PART_ID_TO_INFO)
            partnet_id_info = {
                pid: (cls_name, get_partnet_part_name(cls_name, pid))
                for pid in present
            }
            PART_ID_TO_INFO.update(partnet_id_info)
            class_descs.append(
                generate_description(cls_name, present, cnt, part_xyz)
            )
            agnostic_descs_all.append(
                generate_agnostic_descriptions_all(cls_name, present, cnt, part_xyz)
            )
            PART_ID_TO_INFO.clear()
            PART_ID_TO_INFO.update(_orig)

    return class_descs, agnostic_descs_all
