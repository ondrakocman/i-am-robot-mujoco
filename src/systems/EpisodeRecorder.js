/**
 * EpisodeRecorder.js — 30 Hz data capture for imitation learning
 *
 * Records joint states, actions, object poses, and contact data from
 * the MuJoCo PhysicsBridge at 30 Hz. Episodes are stored in IndexedDB
 * and can be exported as JSON for offline replay and IL training.
 *
 * Data schema matches xr_teleoperate's episode_writer.py format.
 */

const RECORD_HZ = 30
const RECORD_INTERVAL_MS = 1000 / RECORD_HZ
const DB_NAME = 'mujoco_episodes'
const DB_VERSION = 1
const STORE_NAME = 'episodes'

export default class EpisodeRecorder {
  constructor() {
    this.recording = false
    this.frames = []
    this.startTime = 0
    this.frameIdx = 0
    this._intervalId = null
    this._physicsBridge = null
    this._targetAnglesRef = null
    this.db = null
    this.episodes = [] // metadata list
    this._initDB()
  }

  /**
   * Open / create the IndexedDB database.
   */
  async _initDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = (e) => {
        const db = e.target.result
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' })
        }
      }
      req.onsuccess = (e) => {
        this.db = e.target.result
        this._loadEpisodeList().then(resolve)
      }
      req.onerror = (e) => {
        console.error('[EpisodeRecorder] IndexedDB error:', e)
        reject(e)
      }
    })
  }

  async _loadEpisodeList() {
    if (!this.db) return
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly')
      const store = tx.objectStore(STORE_NAME)
      const req = store.getAllKeys()
      req.onsuccess = () => {
        this.episodes = req.result || []
        resolve()
      }
      req.onerror = () => resolve()
    })
  }

  /**
   * Start recording an episode.
   * @param {PhysicsBridge} physicsBridge - The active physics bridge
   * @param {Float64Array} targetAnglesRef - Reference to the 28-element target buffer
   */
  startRecording(physicsBridge, targetAnglesRef) {
    if (this.recording) return
    this._physicsBridge = physicsBridge
    this._targetAnglesRef = targetAnglesRef
    this.frames = []
    this.frameIdx = 0
    this.startTime = performance.now()
    this.recording = true

    this._intervalId = setInterval(() => this._captureFrame(), RECORD_INTERVAL_MS)
    console.log('[EpisodeRecorder] Recording started')
  }

  /**
   * Stop recording and save the episode to IndexedDB.
   * @returns {Promise<string>} The episode ID
   */
  async stopRecording() {
    if (!this.recording) return null
    this.recording = false

    if (this._intervalId !== null) {
      clearInterval(this._intervalId)
      this._intervalId = null
    }

    const id = `ep_${Date.now()}`
    const episode = {
      id,
      info: {
        recorded_at: new Date().toISOString(),
        duration_s: (performance.now() - this.startTime) / 1000,
        num_frames: this.frames.length,
        frequency_hz: RECORD_HZ,
        source: 'mujoco_wasm_quest3',
      },
      data: this.frames,
    }

    await this._saveEpisode(episode)
    this.frames = []
    this._physicsBridge = null
    this._targetAnglesRef = null

    console.log(`[EpisodeRecorder] Saved episode ${id} (${episode.info.num_frames} frames, ${episode.info.duration_s.toFixed(1)}s)`)
    return id
  }

  /**
   * Capture a single frame of data from the physics bridge.
   */
  _captureFrame() {
    if (!this._physicsBridge || !this._physicsBridge.ready) return

    const state = this._physicsBridge.getRawState()
    if (!state) return

    const timestamp = (performance.now() - this.startTime) / 1000
    const targets = this._targetAnglesRef

    const frame = {
      idx: this.frameIdx++,
      timestamp,
      states: {
        left_arm:  { qpos: Array.from(state.qpos.slice(0, 7)),  qvel: Array.from(state.qvel.slice(0, 7)),  torque: Array.from(state.torques.slice(0, 7)) },
        right_arm: { qpos: Array.from(state.qpos.slice(7, 14)), qvel: Array.from(state.qvel.slice(7, 14)), torque: Array.from(state.torques.slice(7, 14)) },
        left_ee:   { qpos: Array.from(state.qpos.slice(14, 21)), qvel: Array.from(state.qvel.slice(14, 21)), torque: Array.from(state.torques.slice(14, 21)) },
        right_ee:  { qpos: Array.from(state.qpos.slice(21, 28)), qvel: Array.from(state.qvel.slice(21, 28)), torque: Array.from(state.torques.slice(21, 28)) },
      },
      actions: {
        left_arm:  { qpos: targets ? Array.from(targets.slice(0, 7))  : [] },
        right_arm: { qpos: targets ? Array.from(targets.slice(7, 14)) : [] },
        left_ee:   { qpos: targets ? Array.from(targets.slice(14, 21)) : [] },
        right_ee:  { qpos: targets ? Array.from(targets.slice(21, 28)) : [] },
      },
      env_state: {
        objects: state.objects.map((obj, i) => ({
          name: ['red_cube', 'blue_cube', 'green_cube'][i] || `object_${i}`,
          pos: Array.from(obj.pos),
          quat: Array.from(obj.quat),
        })),
      },
      contacts: {
        ncon: state.ncon,
      },
    }

    this.frames.push(frame)
  }

  /**
   * Save an episode to IndexedDB.
   */
  async _saveEpisode(episode) {
    if (!this.db) {
      console.warn('[EpisodeRecorder] DB not ready, episode not saved')
      return
    }
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite')
      const store = tx.objectStore(STORE_NAME)
      const req = store.put(episode)
      req.onsuccess = () => {
        this.episodes.push(episode.id)
        resolve()
      }
      req.onerror = (e) => {
        console.error('[EpisodeRecorder] Save error:', e)
        reject(e)
      }
    })
  }

  /**
   * Load an episode from IndexedDB by ID.
   * @param {string} id
   * @returns {Promise<object>}
   */
  async loadEpisode(id) {
    if (!this.db) return null
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly')
      const store = tx.objectStore(STORE_NAME)
      const req = store.get(id)
      req.onsuccess = () => resolve(req.result || null)
      req.onerror = () => resolve(null)
    })
  }

  /**
   * Delete an episode from IndexedDB.
   * @param {string} id
   */
  async deleteEpisode(id) {
    if (!this.db) return
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite')
      const store = tx.objectStore(STORE_NAME)
      store.delete(id)
      tx.oncomplete = () => {
        this.episodes = this.episodes.filter(e => e !== id)
        resolve()
      }
    })
  }

  /**
   * Export an episode as a downloadable JSON file.
   * @param {string} id
   */
  async exportEpisode(id) {
    const episode = await this.loadEpisode(id)
    if (!episode) {
      console.warn(`[EpisodeRecorder] Episode ${id} not found`)
      return
    }
    const json = JSON.stringify(episode, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  /**
   * Export all episodes as a single JSON array download.
   */
  async exportAllEpisodes() {
    if (!this.db) return
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly')
      const store = tx.objectStore(STORE_NAME)
      const req = store.getAll()
      req.onsuccess = () => {
        const all = req.result || []
        const json = JSON.stringify(all, null, 2)
        const blob = new Blob([json], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `all_episodes_${Date.now()}.json`
        a.click()
        URL.revokeObjectURL(url)
        resolve()
      }
    })
  }

  /**
   * Get recording statistics.
   */
  getStats() {
    return {
      recording: this.recording,
      frameCount: this.frames.length,
      duration: this.recording ? (performance.now() - this.startTime) / 1000 : 0,
      episodeCount: this.episodes.length,
      memoryEstimate: this.frames.length * 2, // ~2 KB per frame
    }
  }

  dispose() {
    if (this._intervalId !== null) {
      clearInterval(this._intervalId)
      this._intervalId = null
    }
    this.recording = false
    if (this.db) {
      this.db.close()
      this.db = null
    }
  }
}
