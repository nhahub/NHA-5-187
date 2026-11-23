# Deploying Brain MRI to Azure Container Apps

This is the sequence that worked on an Azure for Students subscription, run from Azure Cloud Shell (Bash). The API and the web UI run as two containers in one Container App, so they talk over `localhost` and share an `/outputs` folder.

**Never commit real secrets.** The registry password and the storage key must not be written into any file in this repo. Replace every `<...>` placeholder below at run time.

## Names used

| Item | Value |
|---|---|
| Resource group | `brainmri-rg` |
| Region | `germanywestcentral` |
| Registry | `brainmriyasmin187` |
| Storage account | `brainmriyasmin187st` |
| File share (model) | `model` |
| Container Apps environment | `brainmri-std-env` |
| Container App | `brainmri-app` |
| Images | `brainmri-api:v1`, `brainmri-web:v1` |

Student subscriptions only allow certain regions. If you get `RequestDisallowedByAzure`, list the allowed ones:

```bash
az policy assignment list --query "[].parameters.listOfAllowedLocations.value" -o json
```

Allowed in our case: spaincentral, francecentral, polandcentral, switzerlandnorth, germanywestcentral.

## 1. Enable providers

```bash
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.OperationalInsights
```

Check they reach `Registered`:

```bash
for p in Microsoft.App Microsoft.ContainerRegistry Microsoft.OperationalInsights; do echo "$p: $(az provider show -n $p --query registrationState -o tsv)"; done
```

## 2. Resource group and registry

```bash
az group create --name brainmri-rg --location westeurope
az acr create --resource-group brainmri-rg --name brainmriyasmin187 --sku Basic --location germanywestcentral
```

The resource group itself can be in any region. The resources inside it must be in an allowed region.

## 3. Build the images in the cloud

Azure pulls the code from GitHub and builds it, so Docker Desktop is not needed. Run one build per image. The API command is the one we ran; the web one is the same pattern with `web/Dockerfile` (confirm it against your own history).

```bash
az acr build --registry brainmriyasmin187 --image brainmri-api:v1 --file api/Dockerfile https://github.com/nhahub/NHA-5-187.git#main
az acr build --registry brainmriyasmin187 --image brainmri-web:v1 --file web/Dockerfile https://github.com/nhahub/NHA-5-187.git#main
```

Check the result:

```bash
az acr repository list --name brainmriyasmin187 -o table
az acr repository show-tags --name brainmriyasmin187 --repository brainmri-api -o tsv
az acr repository show-tags --name brainmriyasmin187 --repository brainmri-web -o tsv
```

## 4. Model files on Azure Files

The model export folder is not in Git (`models/seg-model-v1-torchscript/`). It lives on Google Drive and is uploaded to an Azure file share. The four files must sit at the **root** of the share, not inside a subfolder:

`deploy_manifest.json`, `seg_model.torchscript.pt`, `plans.json`, `dataset.json`

The storage account and share were created in an earlier step whose exact commands are not recorded here. The generic form is:

```bash
az storage account create --name brainmriyasmin187st --resource-group brainmri-rg --location germanywestcentral --sku Standard_LRS
az storage share-rm create --storage-account brainmriyasmin187st --name model --quota 5
```

Verify the contents of the share:

```bash
az storage file list --account-name brainmriyasmin187st --share-name model \
  --account-key "$(az storage account keys list --resource-group brainmri-rg --account-name brainmriyasmin187st --query '[0].value' -o tsv)" \
  --query "[].name" -o tsv
```

## 5. Container Apps environment (workload profiles)

Install or update the extension first. Cloud Shell sessions are temporary, so do this every new session:

```bash
az extension add --name containerapp --upgrade
```

Create the environment with an explicit mode. Without it, an old extension created an Express-type environment, which does not support file share mounts and has a one-per-region limit:

```bash
az containerapp env create --name brainmri-std-env --resource-group brainmri-rg --location germanywestcentral --environment-mode WorkloadProfiles
az containerapp env show --name brainmri-std-env --resource-group brainmri-rg --query "properties.provisioningState" -o tsv
```

Wait for `Succeeded` (up to about 10 minutes). Then attach the file share (the key stays in a shell variable only):

```bash
STORAGE_KEY=$(az storage account keys list --resource-group brainmri-rg --account-name brainmriyasmin187st --query "[0].value" -o tsv)
az containerapp env storage set --name brainmri-std-env --resource-group brainmri-rg --storage-name modelshare --storage-type AzureFile --azure-file-account-name brainmriyasmin187st --azure-file-account-key "$STORAGE_KEY" --azure-file-share-name model --access-mode ReadOnly
```

## 6. Create the app

`az containerapp create --yaml app.yaml` was rejected with a vague `Bad Request ... System.Boolean` error, and we did not find the cause. What worked was creating the app from the CLI first, then applying the YAML with `update`.

Enable registry access and load variables:

```bash
az acr update --name brainmriyasmin187 --admin-enabled true
ACR_PWD=$(az acr credential show --name brainmriyasmin187 --query "passwords[0].value" -o tsv)
ENV_ID=$(az containerapp env show --name brainmri-std-env --resource-group brainmri-rg --query id -o tsv)
```

Create a first revision with the web image only:

```bash
az containerapp create --name brainmri-app --resource-group brainmri-rg --environment brainmri-std-env \
  --image brainmriyasmin187.azurecr.io/brainmri-web:v1 \
  --registry-server brainmriyasmin187.azurecr.io --registry-username brainmriyasmin187 --registry-password "$ACR_PWD" \
  --target-port 8501 --ingress external --cpu 0.5 --memory 1Gi --min-replicas 1 --max-replicas 1
```

Write `app.yaml` in the Cloud Shell in three parts. Do not commit this file, because it contains the registry password after variable substitution.

Part 1 (unquoted `EOF`, so the variables are filled in):

```bash
cat > app.yaml <<EOF
location: germanywestcentral
properties:
  managedEnvironmentId: $ENV_ID
  workloadProfileName: Consumption
  configuration:
    ingress:
      external: true
      targetPort: 8501
      transport: auto
    registries:
    - server: brainmriyasmin187.azurecr.io
      username: brainmriyasmin187
      passwordSecretRef: acr-pwd
    secrets:
    - name: acr-pwd
      value: $ACR_PWD
  template:
    containers:
EOF
```

Part 2 (quoted `'EOF'`, appended):

```bash
cat >> app.yaml <<'EOF'
    - name: api
      image: brainmriyasmin187.azurecr.io/brainmri-api:v1
      resources:
        cpu: 1.5
        memory: 3Gi
      env:
      - name: BRAINMRI_DEVICE
        value: cpu
      - name: API_RUN_RADIOMICS
        value: "true"
      - name: API_TTA
        value: "false"
      - name: API_MAX_UPLOAD_MB
        value: "512"
      - name: API_KEEP_MINUTES
        value: "60"
      volumeMounts:
      - volumeName: models
        mountPath: /models/seg-model-v1-torchscript
      - volumeName: outputs
        mountPath: /outputs
EOF
```

Part 3:

```bash
cat >> app.yaml <<'EOF'
    - name: web
      image: brainmriyasmin187.azurecr.io/brainmri-web:v1
      resources:
        cpu: 0.5
        memory: 1Gi
      env:
      - name: BRAINMRI_API_URL
        value: http://localhost:8000
      volumeMounts:
      - volumeName: outputs
        mountPath: /outputs
    volumes:
    - name: models
      storageType: AzureFile
      storageName: modelshare
    - name: outputs
      storageType: EmptyDir
    scale:
      minReplicas: 1
      maxReplicas: 1
EOF
```

Apply it:

```bash
az containerapp update --name brainmri-app --resource-group brainmri-rg --yaml app.yaml
```

Paste one command block at a time. Pasting a whole terminal transcript back into the shell makes bash try to run the old output as commands.

## 7. Verify

```bash
az containerapp revision list --name brainmri-app --resource-group brainmri-rg --query "[].{name:name,active:properties.active,health:properties.healthState,running:properties.runningState,replicas:properties.replicas}" -o table
az containerapp logs show --name brainmri-app --resource-group brainmri-rg --container api --tail 50
```

Then open the app URL (`https://<app-fqdn>/`):

1. Click **Test Backend Connection**. Expect `Backend is connected!`, with `export_dir_exists: true` and `export_manifest_exists: true`.
2. Upload T1, T1ce, T2 and FLAIR for one case and click **Analyze Scan**.
3. In the API log you should see `POST /predict ... 200 OK`. On CPU our test took about 1.5 minutes. The warning `perform_everything_on_device=True is only supported for cuda devices` is expected on CPU and harmless.

## 8. Cost control

The app holds about 2 CPU and 4 GB of memory while at least one replica runs. To stop that charge but keep everything else (the app wakes on the first request, and that first load is slow):

```bash
az containerapp update --name brainmri-app --resource-group brainmri-rg --min-replicas 0 --max-replicas 1
```

The registry, storage and Log Analytics still bill small fixed amounts. Check **Subscriptions → Cost Management** in the portal.

## 9. Tear down

This permanently deletes the app, registry, images, file share and environment:

```bash
az group delete --name brainmri-rg --yes --no-wait
```

Redeploying afterwards means repeating these steps.

## Security notes

- The registry admin password was printed in a chat session during setup. Rotate it before leaving the deployment running long term: `az acr credential renew --name brainmriyasmin187 --password-name password`, then update the app's registry secret.
- Do not commit `app.yaml`, the storage key, or the `models/` folder.
