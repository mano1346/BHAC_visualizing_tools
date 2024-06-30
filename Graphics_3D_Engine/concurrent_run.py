import asyncio
# pause_event = asyncio.Event()
# pause_event.set()

async def long_running_function(func, args):
    i = 0
    while True:
        await asyncio.to_thread(func, args, i)
        # pause_event.clear()
        # await asyncio.to_thread(app.save_image, i)
        # pause_event.set()
        i +=1

async def other_function_1(func):
    check = True
    while check:
        # await pause_event.wait() 
        await asyncio.sleep(0.0001)
        check = func()


async def main(func1, args1, func2):
    # Create tasks for all functions
    task1 = asyncio.create_task(long_running_function(func1,args1))
    task2 = asyncio.create_task(other_function_1(func2))
    # Wait for the function that checks the condition
    done, pending = await asyncio.wait([task2], return_when=asyncio.FIRST_COMPLETED)

    # Cancel the remaining tasks
    for task in pending:
        task.cancel()

    # Handle cancellation exceptions
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            print(f"{task.get_name()} was cancelled")

    # Optionally wait for the completed task and process the result
    # for task in done:
    #     result = await task
    #     print(result)

    exit()
# Run the main function
def run(func1, args1, func2):
    asyncio.run(main(func1, args1, func2))
