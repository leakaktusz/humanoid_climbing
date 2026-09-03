from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import struct

class Transition(NamedTuple):
    state: jnp.array
    action: jnp.array
    reward: jnp.array
    discount: jnp.array
    next_state: jnp.array

@struct.dataclass
class PureJaxBuffer:
    _states: jnp.array
    _actions: jnp.array
    _next_states: jnp.array
    _rewards: jnp.array
    _discounts: jnp.array
    _ptr: jnp.array
    _size: jnp.array
    _max_size: int = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        state_dim: int,
        action_dim: int,
        max_size: int,
        batch_size: int,
    ) -> "PureJaxBuffer":

        _states = jnp.zeros((max_size, state_dim), dtype=jnp.float32)
        _actions = jnp.zeros((max_size, action_dim), dtype=jnp.float32)
        _next_states = jnp.zeros((max_size, state_dim), dtype=jnp.float32)
        _rewards = jnp.zeros((max_size, 1), dtype=jnp.float32)
        _discounts = jnp.zeros((max_size, 1), dtype=jnp.float32)

        _ptr = jnp.array(0, dtype=jnp.int32)
        _size = jnp.array(0, dtype=jnp.int32)

        return cls(
            _states=_states,
            _actions=_actions,
            _next_states=_next_states,
            _rewards=_rewards,
            _discounts=_discounts,
            _ptr=_ptr,
            _size=_size,
            _max_size=max_size,
            _batch_size=batch_size
        )

    @jax.jit
    def insert(
        self,
        state: jnp.array,
        action: jnp.array,
        reward: jnp.array,
        discount: jnp.array,
        next_state: jnp.array
    ) -> "PureJaxBuffer":

        ptr = self._ptr

        start_idx = (ptr, 0)

        r = jnp.expand_dims(reward, axis=0) if reward.ndim == 0 else reward
        d = jnp.expand_dims(discount, axis=0) if discount.ndim == 0 else discount

        r = jnp.reshape(r, (1, 1))
        d = jnp.reshape(d, (1, 1))

        new_states = jax.lax.dynamic_update_slice(self._states, jnp.expand_dims(state, 0), start_idx)
        new_actions = jax.lax.dynamic_update_slice(self._actions, jnp.expand_dims(action, 0), start_idx)
        new_next_states = jax.lax.dynamic_update_slice(self._next_states, jnp.expand_dims(next_state, 0), start_idx)
        new_rewards = jax.lax.dynamic_update_slice(self._rewards, r, start_idx)
        new_discounts = jax.lax.dynamic_update_slice(self._discounts, d, start_idx)

        new_ptr = (ptr + 1) % self._max_size
        new_size = jnp.minimum(self._size + 1, self._max_size)

        return self.replace(
            _states=new_states,
            _actions=new_actions,
            _next_states=new_next_states,
            _rewards=new_rewards,
            _discounts=new_discounts,
            _ptr=new_ptr,
            _size=new_size
        )

    @jax.jit
    def sample(self, key: jax.random.PRNGKey) -> Transition:
        ind = jax.random.randint(key, (self._batch_size,), 0, self._size)

        r = self._rewards[ind]
        d = self._discounts[ind]

        return Transition(
            state=self._states[ind],
            action=self._actions[ind],
            reward=jnp.squeeze(r),
            discount=jnp.squeeze(d),
            next_state=self._next_states[ind]
        )

    def is_ready(self) -> bool:
        return self._size >= self._batch_size

    @jax.jit
    def insert_batch(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        discount: jnp.ndarray,
        next_state: jnp.ndarray
    ) -> "PureJaxBuffer":
        B = state.shape[0]
        ptr = self._ptr
        max_size = self._max_size

        reward = jnp.reshape(reward, (B, 1))
        discount = jnp.reshape(discount, (B, 1))

        indices = (jnp.arange(B) + ptr) % max_size

        new_states = self._states.at[indices].set(state)
        new_actions = self._actions.at[indices].set(action)
        new_next_states = self._next_states.at[indices].set(next_state)
        new_rewards = self._rewards.at[indices].set(reward)
        new_discounts = self._discounts.at[indices].set(discount)

        new_ptr = (ptr + B) % max_size
        new_size = jnp.minimum(self._size + B, max_size)

        return self.replace(
            _states=new_states,
            _actions=new_actions,
            _next_states=new_next_states,
            _rewards=new_rewards,
            _discounts=new_discounts,
            _ptr=new_ptr,
            _size=new_size
        )
