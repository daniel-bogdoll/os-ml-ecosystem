import fiftyone as fo

def main():
    """Launches FiftyOne app session and waits indefinitely."""
    session = fo.launch_app()
    session.wait(-1)  # (-1) forever

if __name__ == "__main__":
    main()