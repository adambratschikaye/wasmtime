use {
    super::util::{config, make_component},
    anyhow::Result,
    component_async_tests::{Ctx, closed_streams},
    futures::{
        future::{self, FutureExt},
        stream::{FuturesUnordered, StreamExt, TryStreamExt},
    },
    std::{
        future::Future,
        sync::{Arc, Mutex},
    },
    wasmtime::{
        Engine, Store,
        component::{Linker, ResourceTable, StreamReader, StreamWriter, VecBuffer},
    },
    wasmtime_wasi::p2::WasiCtxBuilder,
};

#[tokio::test]
pub async fn async_watch_streams() -> Result<()> {
    let engine = Engine::new(&config())?;

    let mut store = Store::new(
        &engine,
        Ctx {
            wasi: WasiCtxBuilder::new().inherit_stdio().build(),
            table: ResourceTable::default(),
            continue_: false,
            wakers: Arc::new(Mutex::new(None)),
        },
    );

    let mut linker = Linker::new(&engine);

    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;

    let component = make_component(
        &engine,
        &[test_programs_artifacts::ASYNC_CLOSED_STREAMS_COMPONENT],
    )
    .await?;

    let instance = linker.instantiate_async(&mut store, &component).await?;

    // Test watching and then dropping the read end of a stream.
    let (tx, rx) = instance.stream::<u8, Option<_>, Option<_>>(&mut store)?;
    let watch = tx.watch_reader().0;
    drop(rx);
    instance.run(&mut store, watch).await?;

    // Test dropping and then watching the read end of a stream.
    let (tx, rx) = instance.stream::<u8, Option<_>, Option<_>>(&mut store)?;
    drop(rx);
    instance.run(&mut store, tx.watch_reader().0).await?;

    // Test watching and then dropping the write end of a stream.
    let (tx, rx) = instance.stream::<u8, Option<_>, Option<_>>(&mut store)?;
    let watch = rx.watch_writer().0;
    drop(tx);
    instance.run(&mut store, watch).await?;

    // Test dropping and then watching the write end of a stream.
    let (tx, rx) = instance.stream::<u8, Option<_>, Option<_>>(&mut store)?;
    drop(tx);
    instance.run(&mut store, rx.watch_writer().0).await?;

    // Test watching and then dropping the read end of a future.
    let (tx, rx) = instance.future::<u8>(|| 42, &mut store)?;
    let watch = tx.watch_reader().0;
    drop(rx);
    instance.run(&mut store, watch).await?;

    // Test dropping and then watching the read end of a future.
    let (tx, rx) = instance.future::<u8>(|| 42, &mut store)?;
    drop(rx);
    instance.run(&mut store, tx.watch_reader().0).await?;

    // Test watching and then dropping the write end of a future.
    let (tx, rx) = instance.future::<u8>(|| 42, &mut store)?;
    let watch = rx.watch_writer().0;
    drop(tx);
    instance.run(&mut store, watch).await?;

    // Test dropping and then watching the write end of a future.
    let (tx, rx) = instance.future::<u8>(|| 42, &mut store)?;
    drop(tx);
    instance.run(&mut store, rx.watch_writer().0).await?;

    enum Event {
        Write(Option<StreamWriter<Option<u8>>>),
        Read(Option<StreamReader<Option<u8>>>, Option<u8>),
    }

    // Test watching, then writing to, then dropping, then writing again to the
    // read end of a stream.
    let mut futures = FuturesUnordered::new();
    let (tx, rx) = instance.stream(&mut store)?;
    let watch = tx.watch_reader().1;
    futures.push(
        watch
            .into_inner()
            .write_all(Some(42))
            .map(|(w, _)| Event::Write(w))
            .boxed(),
    );
    futures.push(rx.read(None).map(|(r, b)| Event::Read(r, b)).boxed());
    let mut rx = None;
    let mut tx = None;
    while let Some(event) = instance.run(&mut store, futures.next()).await? {
        match event {
            Event::Write(None) => unreachable!(),
            Event::Write(Some(new_tx)) => tx = Some(new_tx),
            Event::Read(None, _) => unreachable!(),
            Event::Read(Some(new_rx), mut buffer) => {
                assert_eq!(buffer.take(), Some(42));
                rx = Some(new_rx);
            }
        }
    }
    drop(rx);
    let (future, watch) = tx.take().unwrap().watch_reader();
    let mut future = Box::pin(future);
    instance
        .run(&mut store, future::poll_fn(|cx| future.as_mut().poll(cx)))
        .await?;
    assert!(
        instance
            .run(&mut store, watch.into_inner().write_all(Some(42)))
            .await?
            .0
            .is_none()
    );

    Ok(())
}

#[tokio::test]
pub async fn async_closed_streams() -> Result<()> {
    test_closed_streams(false).await
}

#[tokio::test]
pub async fn async_closed_streams_with_watch() -> Result<()> {
    test_closed_streams(true).await
}

pub async fn test_closed_streams(watch: bool) -> Result<()> {
    let engine = Engine::new(&config())?;

    let mut store = Store::new(
        &engine,
        Ctx {
            wasi: WasiCtxBuilder::new().inherit_stdio().build(),
            table: ResourceTable::default(),
            continue_: false,
            wakers: Arc::new(Mutex::new(None)),
        },
    );

    let mut linker = Linker::new(&engine);

    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;

    let component = make_component(
        &engine,
        &[test_programs_artifacts::ASYNC_CLOSED_STREAMS_COMPONENT],
    )
    .await?;

    let instance = linker.instantiate_async(&mut store, &component).await?;

    let closed_streams = closed_streams::bindings::ClosedStreams::new(&mut store, &instance)?;

    enum StreamEvent {
        FirstWrite(Option<StreamWriter<VecBuffer<u8>>>),
        FirstRead(Option<StreamReader<Vec<u8>>>, Vec<u8>),
        SecondWrite(Option<StreamWriter<VecBuffer<u8>>>),
        GuestCompleted,
    }

    enum FutureEvent {
        Write(bool),
        Read(Option<u8>),
        WriteIgnored(bool),
        GuestCompleted,
    }

    let values = vec![42_u8, 43, 44];

    let value = 42_u8;

    // First, test stream host->host
    {
        let (tx, rx) = instance.stream(&mut store)?;

        let mut futures = FuturesUnordered::new();
        futures.push(
            tx.write_all(values.clone().into())
                .map(|(w, _)| StreamEvent::FirstWrite(w))
                .boxed(),
        );
        futures.push(
            rx.read(Vec::with_capacity(3))
                .map(|(r, b)| StreamEvent::FirstRead(r, b))
                .boxed(),
        );

        let mut count = 0;
        while let Some(event) = instance.run(&mut store, futures.next()).await? {
            count += 1;
            match event {
                StreamEvent::FirstWrite(Some(tx)) => {
                    if watch {
                        futures.push(
                            tx.watch_reader()
                                .0
                                .map(|()| StreamEvent::SecondWrite(None))
                                .boxed(),
                        );
                    } else {
                        futures.push(
                            tx.write_all(values.clone().into())
                                .map(|(w, _)| StreamEvent::SecondWrite(w))
                                .boxed(),
                        );
                    }
                }
                StreamEvent::FirstWrite(None) => panic!("first write should have been accepted"),
                StreamEvent::FirstRead(Some(_), results) => {
                    assert_eq!(values, results);
                }
                StreamEvent::FirstRead(None, _) => unreachable!(),
                StreamEvent::SecondWrite(None) => {}
                StreamEvent::SecondWrite(Some(_)) => {
                    panic!("second write should _not_ have been accepted")
                }
                StreamEvent::GuestCompleted => unreachable!(),
            }
        }

        assert_eq!(count, 3);
    }

    // Next, test futures host->host
    {
        let (tx, rx) = instance.future(|| unreachable!(), &mut store)?;
        let (tx_ignored, rx_ignored) = instance.future(|| 42u8, &mut store)?;

        let mut futures = FuturesUnordered::new();
        futures.push(tx.write(value).map(FutureEvent::Write).boxed());
        futures.push(rx.read().map(FutureEvent::Read).boxed());
        if watch {
            futures.push(
                tx_ignored
                    .watch_reader()
                    .0
                    .map(|()| FutureEvent::WriteIgnored(false))
                    .boxed(),
            );
        } else {
            futures.push(
                tx_ignored
                    .write(value)
                    .map(FutureEvent::WriteIgnored)
                    .boxed(),
            );
        }
        drop(rx_ignored);

        let mut count = 0;
        while let Some(event) = instance.run(&mut store, futures.next()).await? {
            count += 1;
            match event {
                FutureEvent::Write(delivered) => {
                    assert!(delivered);
                }
                FutureEvent::Read(Some(result)) => {
                    assert_eq!(value, result);
                }
                FutureEvent::Read(None) => panic!("read should have succeeded"),
                FutureEvent::WriteIgnored(delivered) => {
                    assert!(!delivered);
                }
                FutureEvent::GuestCompleted => unreachable!(),
            }
        }

        assert_eq!(count, 3);
    }

    // Next, test stream host->guest
    {
        let (tx, rx) = instance.stream::<_, _, Vec<_>>(&mut store)?;

        let mut futures = FuturesUnordered::new();
        futures.push(
            closed_streams
                .local_local_closed()
                .call_read_stream(&mut store, rx.into(), values.clone())
                .map(|v| v.map(|()| StreamEvent::GuestCompleted))
                .boxed(),
        );
        futures.push(
            tx.write_all(values.clone().into())
                .map(|(w, _)| Ok(StreamEvent::FirstWrite(w)))
                .boxed(),
        );

        let mut count = 0;
        while let Some(event) = instance.run(&mut store, futures.try_next()).await?? {
            count += 1;
            match event {
                StreamEvent::FirstWrite(Some(tx)) => {
                    if watch {
                        futures.push(
                            tx.watch_reader()
                                .0
                                .map(|()| Ok(StreamEvent::SecondWrite(None)))
                                .boxed(),
                        );
                    } else {
                        futures.push(
                            tx.write_all(values.clone().into())
                                .map(|(w, _)| Ok(StreamEvent::SecondWrite(w)))
                                .boxed(),
                        );
                    }
                }
                StreamEvent::FirstWrite(None) => panic!("first write should have been accepted"),
                StreamEvent::FirstRead(_, _) => unreachable!(),
                StreamEvent::SecondWrite(None) => {}
                StreamEvent::SecondWrite(Some(_)) => {
                    panic!("second write should _not_ have been accepted")
                }
                StreamEvent::GuestCompleted => {}
            }
        }

        assert_eq!(count, 3);
    }

    // Next, test futures host->guest
    {
        let (tx, rx) = instance.future(|| unreachable!(), &mut store)?;
        let (tx_ignored, rx_ignored) = instance.future(|| 0, &mut store)?;

        let mut futures = FuturesUnordered::new();
        futures.push(
            closed_streams
                .local_local_closed()
                .call_read_future(&mut store, rx.into(), value, rx_ignored.into())
                .map(|v| v.map(|()| FutureEvent::GuestCompleted))
                .boxed(),
        );
        futures.push(tx.write(value).map(FutureEvent::Write).map(Ok).boxed());
        if watch {
            futures.push(
                tx_ignored
                    .watch_reader()
                    .0
                    .map(|()| Ok(FutureEvent::WriteIgnored(false)))
                    .boxed(),
            );
        } else {
            futures.push(
                tx_ignored
                    .write(value)
                    .map(FutureEvent::WriteIgnored)
                    .map(Ok)
                    .boxed(),
            );
        }

        let mut count = 0;
        while let Some(event) = instance.run(&mut store, futures.try_next()).await?? {
            count += 1;
            match event {
                FutureEvent::Write(delivered) => {
                    assert!(delivered);
                }
                FutureEvent::Read(_) => unreachable!(),
                FutureEvent::WriteIgnored(delivered) => {
                    assert!(!delivered);
                }
                FutureEvent::GuestCompleted => {}
            }
        }

        assert_eq!(count, 3);
    }

    Ok(())
}
