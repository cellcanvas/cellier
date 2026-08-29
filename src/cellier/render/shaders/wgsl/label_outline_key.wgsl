// ── Screen-space outline key ──────────────────────────────────────────────
// Included by every cellier label shader.  Produces the 32-bit value written
// to the `outline_id` render target, which the outline composite pass
// compares between neighbouring pixels to find label boundaries.
//
// The field is **range-partitioned**, which is what makes selection exact
// rather than merely very likely:
//
//   0        background / unlabelled  — never outlined
//   1..15    selected label, palette slot v - 1
//   16..     unselected label, 16 + hash
//
// Because the ranges cannot overlap, no unselected label can ever be
// mistaken for a selected one.  Two selected labels stay distinguishable
// from each other, so touching selected labels keep their shared boundary.
//
// Requires from the including shader:
//   u_label_params        LabelParams uniform (background_label, salt,
//                         n_entries, n_outline_entries)
//   t_outline_selection   texture_2d<i32>, rg32sint — sorted (label, slot)
//                         pairs.  Bound only when has_outline_selection.

// The mixing function behind the random colormap.  `random_label_color`
// calls this too, so a key collision implies a hue collision and the
// (vanishingly rare) missing boundary sits between two identically coloured
// regions rather than reading as a bug.
fn random_label_hash(label_id: i32, salt: u32) -> u32 {
    var x = bitcast<u32>(label_id) ^ salt;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    return x;
}

$$ if has_outline_selection
// Binary search the sorted (label_id, slot) pairs.  Same shape as the
// direct colormap's lookup; 17 iterations covers 65536 entries.
fn outline_selection_slot(label_id: i32) -> u32 {
    var lo = 0;
    var hi = i32(u_label_params.n_outline_entries);
    for (var iter = 0; iter < 17; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2;
        let entry = textureLoad(t_outline_selection, vec2<i32>(mid, 0), 0);
        if (entry.r == label_id) {
            return u32(clamp(entry.g, 1, 15));
        }
        if (entry.r < label_id) { lo = mid + 1; } else { hi = mid; }
    }
    return 0u;
}
$$ endif

fn label_outline_key(label_id: i32) -> u32 {
    if (label_id == u_label_params.background_label) {
        return 0u;
    }
    $$ if has_outline_selection
    let slot = outline_selection_slot(label_id);
    if (slot != 0u) {
        return slot;
    }
    $$ endif
    // Mask to 28 bits and offset past the selection range.  That leaves
    // 2^28 buckets: a label sharing a key with any of ~8 screen neighbours
    // has probability about 3e-8.
    return 16u + (random_label_hash(label_id, u_label_params.salt) & 0x0FFFFFFFu);
}
