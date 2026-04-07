//! Event propagation — pluggable dispatch strategies.
//!
//! The default strategy mirrors the W3C Capture → Target → Bubble model,
//! but any tree-based system can define its own strategy:
//!
//! - **DOM** uses capture → target → bubble (web standard)
//! - **3D scene** might use ray-cast → nearest hit → parent bubbling
//! - **JS engine** might wrap DOM propagation with script callbacks
//!
//! ## Design
//!
//! [`PropagationStrategy`] is defined once here; downstream crates implement it.
//! The generic `Arena<D>` in [`tree`](crate::tree) doesn't hard-code any
//! dispatch — the host calls the strategy with the arena + event.

use crate::interaction::{EventContext, EventResponse, InputEvent, Phase};
use crate::tree::{Arena, NodeId};

// Re-export from interaction — single source of truth.
pub use crate::interaction::DispatchResult;

// ═══════════════════════════════════════════════════════════════════════════
// ── Propagation strategy trait ──────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Pluggable event dispatch strategy for tree-based systems.
///
/// Implementations decide HOW an event travels through the tree:
/// - Which nodes see the event
/// - In what order (capture, bubble, or something custom)
/// - When to stop propagation
///
/// The generic parameter `D` matches `Arena<D>`, keeping the strategy
/// decoupled from any specific framework's node data.
pub trait PropagationStrategy<D> {
    /// Dispatch an event into the arena starting from `target`.
    ///
    /// The implementation should:
    /// 1. Build the ancestor path
    /// 2. Walk nodes in the desired order
    /// 3. Call node-specific handlers (via the `handler` callback)
    /// 4. Respect `ctx.stopped` to halt propagation
    fn dispatch(
        &self,
        arena: &mut Arena<D>,
        target: NodeId,
        event: InputEvent,
        handler: &mut dyn FnMut(&mut D, &mut EventContext) -> EventResponse,
    ) -> DispatchResult;
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Capture → Target → Bubble (default) ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Standard W3C-style event propagation: capture → target → bubble.
///
/// This is the default strategy used by DOM trees. Other systems can
/// implement [`PropagationStrategy`] with different traversal orders.
pub struct CaptureBubble;

impl<D> PropagationStrategy<D> for CaptureBubble {
    fn dispatch(
        &self,
        arena: &mut Arena<D>,
        target: NodeId,
        event: InputEvent,
        handler: &mut dyn FnMut(&mut D, &mut EventContext) -> EventResponse,
    ) -> DispatchResult {
        let path = arena.ancestor_path(target);
        let tags = arena.collect_tags(&path);
        let mut ctx = EventContext::new(event);

        // Capture phase: root → target-1
        ctx.phase = Phase::Capture;
        for &id in &path[..path.len().saturating_sub(1)] {
            if ctx.stopped {
                break;
            }
            handler(&mut arena.node_mut(id).data, &mut ctx);
        }

        // Target phase
        if !ctx.stopped {
            ctx.phase = Phase::Target;
            handler(&mut arena.node_mut(target).data, &mut ctx);
        }

        // Bubble phase: target-1 → root (reverse)
        if !ctx.stopped {
            ctx.phase = Phase::Bubble;
            for &id in path.iter().rev().skip(1) {
                if ctx.stopped {
                    break;
                }
                handler(&mut arena.node_mut(id).data, &mut ctx);
            }
        }

        DispatchResult {
            tags,
            stopped: ctx.stopped,
            default_prevented: ctx.default_prevented,
            restyled: false,
            cursor: String::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Immediate (no propagation) ──────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// No propagation — event goes directly to the target node only.
///
/// Suitable for 3D scenes, game objects, or systems where events
/// don't need to traverse a hierarchy.
pub struct Immediate;

impl<D> PropagationStrategy<D> for Immediate {
    fn dispatch(
        &self,
        arena: &mut Arena<D>,
        target: NodeId,
        event: InputEvent,
        handler: &mut dyn FnMut(&mut D, &mut EventContext) -> EventResponse,
    ) -> DispatchResult {
        let mut ctx = EventContext::new(event);
        ctx.phase = Phase::Target;
        handler(&mut arena.node_mut(target).data, &mut ctx);

        let tag = arena.node(target).tag.clone();
        DispatchResult {
            tags: tag.into_iter().collect(),
            stopped: ctx.stopped,
            default_prevented: ctx.default_prevented,
            restyled: false,
            cursor: String::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::InputEvent;

    #[test]
    fn capture_bubble_dispatch() {
        let mut arena = Arena::new(());
        arena.tag(arena.root, "root");
        let child = arena.add(arena.root, ());
        arena.tag(child, "child");

        let mut phases_seen = Vec::new();
        let result =
            CaptureBubble.dispatch(&mut arena, child, InputEvent::Focus, &mut |_data, ctx| {
                phases_seen.push(ctx.phase);
                EventResponse::Ignored
            });

        // Capture (root) → Target (child) → Bubble (root)
        assert_eq!(
            phases_seen,
            vec![Phase::Capture, Phase::Target, Phase::Bubble]
        );
        assert_eq!(result.tags, vec!["root".to_string(), "child".to_string()]);
    }

    #[test]
    fn immediate_dispatch() {
        let mut arena = Arena::new(());
        let child = arena.add(arena.root, ());
        arena.tag(child, "target");

        let mut call_count = 0;
        let result =
            Immediate.dispatch(&mut arena, child, InputEvent::Focus, &mut |_data, _ctx| {
                call_count += 1;
                EventResponse::Consumed
            });

        assert_eq!(call_count, 1);
        assert_eq!(result.tags, vec!["target".to_string()]);
    }

    #[test]
    fn stop_propagation_halts_bubble() {
        let mut arena = Arena::new(());
        let child = arena.add(arena.root, ());
        let grandchild = arena.add(child, ());

        let mut visits = 0;
        let result = CaptureBubble.dispatch(
            &mut arena,
            grandchild,
            InputEvent::Focus,
            &mut |_data, ctx| {
                visits += 1;
                if ctx.phase == Phase::Target {
                    ctx.stop_propagation();
                }
                EventResponse::Ignored
            },
        );

        // Capture(root) + Capture(child) + Target(grandchild) = 3, no bubble
        assert_eq!(visits, 3);
        assert!(result.stopped);
    }
}
