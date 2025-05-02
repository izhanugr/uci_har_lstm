import argparse
import train
import test 

def main():
    parser = argparse.ArgumentParser(description="UCI HAR LSTM Classifier CLI")
    subparser = parser.add_subparsers(dest="command")

    #train command
    train_parser = subparser.add_parser("train", help="Train the LSTM model")
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--num_layer", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--bidirectional", type=bool, default=False)
    

    #test command
    test_parser = subparser.add_parser("test", help="Evaluate the model")

    test_parser.add_argument("--num_layer", type=int, default=1)
    test_parser.add_argument("--dropout", type=float, default=0.0)
    test_parser.add_argument("--bidirectional", type=bool, default=False)

    args = parser.parse_args()

    if args.command == "train":
        train.run_training(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            num_layer=args.num_layer,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif args.command == "test":
        test.run_test(
            num_layer=args.num_layer,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()