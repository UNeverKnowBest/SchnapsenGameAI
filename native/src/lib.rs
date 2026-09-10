mod features;
use features::{ACTION_DIM, FEATURE_VERSION, STATE_DIM, decode_action, encode};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, ndarray::Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use schnapsen_engine::{Game, evaluation::Random};

type BatchOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<u8>>,
);
struct Row {
    state: [f32; STATE_DIM],
    mask: [u8; ACTION_DIM],
    player: i8,
    winner: i8,
    points: u8,
}
#[pyclass]
struct BatchEnv {
    games: Vec<Game>,
    pool: rayon::ThreadPool,
}
impl BatchEnv {
    fn rows(&self) -> Vec<Row> {
        self.pool.install(|| {
            self.games
                .par_iter()
                .map(|game| {
                    if let Some(outcome) = game.outcome() {
                        Row {
                            state: [0.; STATE_DIM],
                            mask: [0; ACTION_DIM],
                            player: -1,
                            winner: outcome.winner.index() as i8,
                            points: outcome.game_points,
                        }
                    } else {
                        let player = game.current_player().unwrap();
                        let (state, mask) = encode(&game.observation_recent(player, 5));
                        Row {
                            state,
                            mask,
                            player: player.index() as i8,
                            winner: -1,
                            points: 0,
                        }
                    }
                })
                .collect()
        })
    }
}
fn arrays(py: Python<'_>, rows: Vec<Row>) -> BatchOutput<'_> {
    let n = rows.len();
    let mut states = Vec::with_capacity(n * STATE_DIM);
    let mut masks = Vec::with_capacity(n * ACTION_DIM);
    let mut players = Vec::with_capacity(n);
    let mut winners = Vec::with_capacity(n);
    let mut points = Vec::with_capacity(n);
    for row in rows {
        states.extend_from_slice(&row.state);
        masks.extend_from_slice(&row.mask);
        players.push(row.player);
        winners.push(row.winner);
        points.push(row.points);
    }
    (
        Array2::from_shape_vec((n, STATE_DIM), states)
            .unwrap()
            .into_pyarray(py),
        Array2::from_shape_vec((n, ACTION_DIM), masks)
            .unwrap()
            .into_pyarray(py),
        players.into_pyarray(py),
        winners.into_pyarray(py),
        points.into_pyarray(py),
    )
}
#[pymethods]
impl BatchEnv {
    #[new]
    fn new(num_envs: usize, workers: usize) -> PyResult<Self> {
        if num_envs == 0 || workers == 0 {
            return Err(PyValueError::new_err(
                "num_envs and workers must be positive",
            ));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let games = (0..num_envs)
            .map(|_| Game::from_deck(schnapsen_engine::Card::deck()).unwrap())
            .collect();
        Ok(Self { games, pool })
    }
    #[pyo3(signature = (seed, first_game, paired=false))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: u64,
        first_game: u64,
        paired: bool,
    ) -> BatchOutput<'py> {
        let rows = py.detach(|| {
            self.pool.install(|| {
                self.games
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(lane, game)| {
                        let id = first_game.wrapping_add(lane as u64);
                        let id = if paired { id / 2 } else { id };
                        let mut rng =
                            Random::new(seed.wrapping_add(id.wrapping_mul(0x9e3779b97f4a7c15)));
                        *game = Game::from_deck(rng.deck()).unwrap();
                    })
            });
            self.rows()
        });
        arrays(py, rows)
    }
    /// One action per lane; -1 is required for finished lanes.
    /// Validate the whole batch first: a rejected batch never partially advances.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<BatchOutput<'py>> {
        let ids = actions.as_slice()?.to_vec();
        if ids.len() != self.games.len() {
            return Err(PyValueError::new_err("wrong action batch length"));
        }
        let rows = py
            .detach(|| -> Result<Vec<Row>, String> {
                let moves: Result<Vec<_>, String> = self
                    .games
                    .iter()
                    .zip(&ids)
                    .map(|(game, &id)| {
                        if game.outcome().is_some() {
                            if id != -1 {
                                return Err("finished lane requires action -1".into());
                            }
                            Ok(None)
                        } else {
                            let action = decode_action(id as usize)?;
                            if !game.legal_moves().contains(&action) {
                                return Err(format!("illegal action {id}"));
                            }
                            Ok(Some(action))
                        }
                    })
                    .collect();
                let moves = moves?;
                self.pool.install(|| {
                    self.games
                        .par_iter_mut()
                        .zip(moves.par_iter())
                        .for_each(|(game, action)| {
                            if let Some(action) = action {
                                game.step(*action).expect("batch validated");
                            }
                        })
                });
                Ok(self.rows())
            })
            .map_err(PyValueError::new_err)?;
        Ok(arrays(py, rows))
    }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchEnv>()?;
    m.add("STATE_DIM", STATE_DIM)?;
    m.add("ACTION_DIM", ACTION_DIM)?;
    m.add("FEATURE_VERSION", FEATURE_VERSION)?;
    Ok(())
}
