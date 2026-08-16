// Overlay UI. Plain DOM -- the interesting code is in the compiler, and a
// framework here would be the largest dependency in the repository.

import { CODEX } from './data/codex.js';
import { decodeContext } from './world.js';
import { explain } from './theme.js';

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export class Hud {
  constructor(root, world, callbacks) {
    this.root = root;
    this.world = world;
    this.callbacks = callbacks;
    this.theme = null;
    this.build();
  }

  build() {
    const header = el('header', 'panel panel-header');
    header.append(el('h1', null, 'lidarworld'));
    const sub = el('p', 'muted');
    sub.textContent = `${this.world.header.name} · ${this.world.header.summary.nodes} nodes · `
      + `${this.world.indices.length / 3 | 0} triangles`;
    header.append(sub);
    if (this.world.header.crs) header.append(el('p', 'muted tiny', this.world.header.crs));

    // --- themes ---------------------------------------------------------
    const themePanel = el('section', 'panel');
    themePanel.append(el('h2', null, 'Theme'));
    themePanel.append(el('p', 'muted tiny',
      'Same geometry, different rule pack. Nothing is re-meshed.'));
    this.themeButtons = el('div', 'row wrap');
    for (const id of this.world.header.themes) {
      const button = el('button', 'chip', id);
      button.dataset.theme = id;
      button.addEventListener('click', () => this.callbacks.onTheme(id));
      this.themeButtons.append(button);
    }
    themePanel.append(this.themeButtons);
    this.themeNote = el('p', 'muted tiny');
    themePanel.append(this.themeNote);

    // --- view -----------------------------------------------------------
    const viewPanel = el('section', 'panel');
    viewPanel.append(el('h2', null, 'View'));
    this.debugSelect = el('select');
    [['0', 'Themed materials'], ['1', 'Context flags'], ['2', 'Roles'],
     ['3', 'Measured vs inferred']].forEach(([value, label]) => {
      const option = el('option', null, label);
      option.value = value;
      this.debugSelect.append(option);
    });
    this.debugSelect.addEventListener('change', () =>
      this.callbacks.onDebugMode(Number(this.debugSelect.value)));
    viewPanel.append(this.debugSelect);

    const toggles = el('div', 'row wrap');
    this.toggle(toggles, 'Source points', false, (on) => this.callbacks.onPoints(on));
    this.toggle(toggles, 'Props', true, (on) => this.callbacks.onInstances(on));
    this.toggle(toggles, 'Fly', false, (on) => this.callbacks.onFly(on));
    viewPanel.append(toggles);

    viewPanel.append(el('h3', null, 'Highlight context flag'));
    this.flagSelect = el('select');
    const none = el('option', null, '— none —');
    none.value = '0';
    this.flagSelect.append(none);
    for (const [name, bit] of Object.entries(this.world.contextFlags)) {
      const option = el('option', null, name);
      option.value = String(bit);
      this.flagSelect.append(option);
    }
    this.flagSelect.addEventListener('change', () =>
      this.callbacks.onContextMask(Number(this.flagSelect.value)));
    viewPanel.append(this.flagSelect);

    // --- inspector ------------------------------------------------------
    this.inspector = el('section', 'panel inspector');
    this.inspector.append(el('h2', null, 'Inspector'));
    this.inspectorBody = el('div');
    this.inspectorBody.append(el('p', 'muted tiny',
      'Aim the crosshair at a surface and press E (or click while unlocked).'));
    this.inspector.append(this.inspectorBody);

    // --- resources ------------------------------------------------------
    const resources = el('section', 'panel collapsed');
    const title = el('h2', 'clickable', `Resources (${CODEX.length})`);
    const list = el('div', 'codex hidden');
    title.addEventListener('click', () => list.classList.toggle('hidden'));
    const groups = new Map();
    for (const entry of CODEX) {
      if (!groups.has(entry.g)) groups.set(entry.g, []);
      groups.get(entry.g).push(entry);
    }
    for (const [group, entries] of groups) {
      list.append(el('h3', null, group));
      for (const entry of entries) {
        const link = el('a', 'codex-item');
        link.href = entry.u;
        link.target = '_blank';
        link.rel = 'noreferrer noopener';
        link.append(el('strong', null, entry.n));
        link.append(el('span', 'muted tiny', ` ${entry.d}`));
        list.append(link);
      }
    }
    resources.append(title, list);
    resources.append(el('p', 'muted tiny',
      'Index built from szenergy/awesome-lidar.'));

    this.left = el('div', 'column left');
    this.left.append(header, themePanel, viewPanel, resources);
    this.right = el('div', 'column right');
    this.right.append(this.inspector);

    this.stats = el('div', 'stats');
    this.crosshair = el('div', 'crosshair');
    const help = el('div', 'help');
    help.innerHTML = 'click to look · <b>WASD</b> move · <b>shift</b> sprint · '
      + '<b>space</b> jump/up · <b>E</b> inspect · <b>F</b> fly · <b>P</b> points · <b>R</b> reset';

    this.root.append(this.left, this.right, this.stats, this.crosshair, help);
  }

  toggle(parent, label, initial, onChange) {
    const button = el('button', `chip${initial ? ' active' : ''}`, label);
    let state = initial;
    button.addEventListener('click', () => {
      state = !state;
      button.classList.toggle('active', state);
      onChange(state);
    });
    parent.append(button);
    return {
      set(value) {
        state = value;
        button.classList.toggle('active', state);
      },
    };
  }

  setTheme(theme, info) {
    this.theme = theme;
    for (const button of this.themeButtons.children) {
      button.classList.toggle('active', button.dataset.theme === theme.id);
    }
    this.themeNote.textContent =
      `${theme.name} · ${theme.materials.length} materials · ${theme.rules.length} rules · `
      + `${info.distinctRequests} distinct (role, context) requests resolved`;
  }

  setStats(text) {
    this.stats.textContent = text;
  }

  showPick(pick) {
    const body = this.inspectorBody;
    body.replaceChildren();
    if (!pick) {
      body.append(el('p', 'muted tiny', 'Nothing under the crosshair.'));
      return;
    }

    const node = pick.nodeId ? this.world.graph.get(pick.nodeId) : null;
    const flags = decodeContext(this.world.contextFlags, pick.context);

    const table = el('dl');
    const row = (key, value) => {
      table.append(el('dt', null, key));
      table.append(el('dd', null, value));
    };
    row('role', pick.role);
    if (node) {
      row('node', node.id);
      row('semantic', node.semantic);
      row('confidence', node.confidence.toFixed(2));
      if (node.support) row('support', `${node.support} points`);
      if (node.parent) row('parent', node.parent);
      for (const [key, value] of Object.entries(node.attrs || {})) {
        if (key === 'context') continue;
        row(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
      }
    }
    row('distance', `${pick.distance.toFixed(2)} m`);
    body.append(table);

    body.append(el('h3', null, `context (${flags.length})`));
    const chips = el('div', 'row wrap');
    for (const flag of flags) chips.append(el('span', 'flag', flag));
    if (!flags.length) chips.append(el('span', 'muted tiny', 'none set'));
    body.append(chips);

    if (this.theme) {
      const resolved = explain(this.theme, pick.role, pick.context);
      body.append(el('h3', null, 'material resolution'));
      const detail = el('dl');
      const add = (key, value) => {
        detail.append(el('dt', null, key));
        detail.append(el('dd', null, value));
      };
      add('material', resolved.spec ? resolved.spec.id : '—');
      if (resolved.rule) {
        const predicate = [
          resolved.rule.role !== '*' ? `role ${resolved.rule.role}` : null,
          resolved.rule.all ? `all(${decodeContext(this.world.contextFlags, resolved.rule.all).join(', ')})` : null,
          resolved.rule.any ? `any(${decodeContext(this.world.contextFlags, resolved.rule.any).join(', ')})` : null,
          resolved.rule.none ? `none(${decodeContext(this.world.contextFlags, resolved.rule.none).join(', ')})` : null,
        ].filter(Boolean).join(' · ');
        add('matched rule', predicate || 'default');
        if (resolved.rule.note) add('why', resolved.rule.note);
      } else {
        add('matched rule', 'fallback');
      }
      if (resolved.spec) {
        const provenance = resolved.spec.provenance || {};
        add('source', provenance.generator
          ? `procedural: ${provenance.generator}` : (resolved.spec.source || '—'));
        add('licence', resolved.spec.license || 'unknown');
        add('tile scale', `${resolved.spec.scale} m`);
      }
      body.append(detail);
    }

    const relations = this.world.edges.filter((e) => e.a === pick.nodeId || e.b === pick.nodeId);
    if (relations.length) {
      body.append(el('h3', null, `relations (${relations.length})`));
      const list = el('ul', 'relations');
      for (const edge of relations.slice(0, 12)) {
        const other = edge.a === pick.nodeId ? edge.b : edge.a;
        list.append(el('li', null, `${edge.rel} → ${other} (${edge.confidence.toFixed(2)})`));
      }
      body.append(list);
    }
  }

  setError(message) {
    const box = el('section', 'panel error');
    box.append(el('h2', null, 'Could not load the world'));
    box.append(el('p', null, message));
    box.append(el('p', 'muted tiny',
      'Compile one first:  lidarworld compile data/samples/townblock.las '
      + '-o viewer/world --theme survey --theme victorian --theme neon'));
    this.left.replaceChildren(box);
  }
}
