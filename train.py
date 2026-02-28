

# Training loop

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    thresholds = np.array([0.5]*y.shape[1])
    preds_bin = (all_preds > thresholds).astype(int)
    micro_f1 = f1_score(all_targets, preds_bin, average='micro')
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Micro-F1: {micro_f1:.4f}")


# Final evaluation

preds_bin = (all_preds > thresholds).astype(int)
print("FINAL MICRO F1:", f1_score(all_targets, preds_bin, average='micro'))
print(classification_report(all_targets, preds_bin))


# Save model and thresholds

torch.save(model.state_dict(), "stutter_nn_weights.pth")
torch.save(model, "stutter_nn_full.pth")
np.save("optimal_thresholds.npy", thresholds)
print("✅ Model, weights, and thresholds saved.")
