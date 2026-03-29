bl_info = {
    "name": "Alpha Blend Effect",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "category": "Sequencer",
    "description": "Blend two video strips: alpha * A + (1 - alpha) * B",
}

import bpy


def _get_strips(scene):
    """Return (sequence_editor, strips_collection) with Blender 4/5 compat."""
    sed = scene.sequence_editor
    if sed is None:
        return None, []
    return sed, getattr(sed, 'strips', None) or getattr(sed, 'sequences', [])


def _active_effect(context):
    """Return the active strip if it is one of our Alpha Blend effects."""
    sed = context.scene.sequence_editor
    if sed is None:
        return None
    active = getattr(sed, 'active_strip', None)
    if active and active.type == 'GAMMA_CROSS' and active.name.startswith("Alpha Blend"):
        return active
    return None


# --- Operator ----------------------------------------------------------------

class SEQUENCER_OT_alpha_blend(bpy.types.Operator):
    """Create an Alpha Blend crossfade between two selected video strips"""
    bl_idname = "sequencer.alpha_blend"
    bl_label = "Alpha Blend"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sed, strips = _get_strips(context.scene)
        if sed is None:
            sed = context.scene.sequence_editor_create()
            _, strips = _get_strips(context.scene)

        selected = sorted(
            (s for s in strips if s.select and s.type != 'SOUND'),
            key=lambda s: s.channel,
        )
        if len(selected) < 2:
            self.report({'ERROR'}, f"Select at least 2 video strips (got {len(selected)})")
            return {'CANCELLED'}

        strip_a, strip_b = selected[0], selected[1]
        start = max(strip_a.frame_final_start, strip_b.frame_final_start)
        end = min(strip_a.frame_final_end, strip_b.frame_final_end)
        if end <= start:
            self.report({'ERROR'}, "Strips must overlap in time")
            return {'CANCELLED'}

        effect = strips.new_effect(
            name="Alpha Blend",
            type='GAMMA_CROSS',
            channel=max(strip_a.channel, strip_b.channel) + 1,
            frame_start=start,
            length=end - start,
            input1=strip_a,
            input2=strip_b,
        )
        effect.use_default_fade = False
        effect.effect_fader = 0.5

        sed.active_strip = effect
        return {'FINISHED'}


# --- Panel -------------------------------------------------------------------

class SEQUENCER_PT_alpha_blend(bpy.types.Panel):
    bl_label = "Alpha Blend"
    bl_space_type = 'SEQUENCE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Alpha Blend"

    def draw(self, context):
        layout = self.layout
        _, strips = _get_strips(context.scene)
        effect = _active_effect(context)

        # List all Alpha Blend effects so user can see they are separate
        effects = [s for s in strips
                   if s.type == 'GAMMA_CROSS' and s.name.startswith("Alpha Blend")]

        if effects:
            box = layout.box()
            box.label(text="Effects:")
            for e in effects:
                row = box.row(align=True)
                icon = 'RESTRICT_SELECT_OFF' if (e == effect) else 'RESTRICT_SELECT_ON'
                row.label(text=e.name, icon=icon)
                row.prop(e, "effect_fader", text="", slider=True)

        if effect:
            layout.separator()
            layout.label(text=f"Active: {effect.name}")
            layout.label(text="Click a strip in the timeline to switch")

        layout.separator()
        layout.operator("sequencer.alpha_blend", text="New Alpha Blend")


# --- Register / Unregister ---------------------------------------------------

def register():
    bpy.utils.register_class(SEQUENCER_OT_alpha_blend)
    bpy.utils.register_class(SEQUENCER_PT_alpha_blend)


def unregister():
    bpy.utils.unregister_class(SEQUENCER_PT_alpha_blend)
    bpy.utils.unregister_class(SEQUENCER_OT_alpha_blend)


if __name__ == "__main__":
    register()
