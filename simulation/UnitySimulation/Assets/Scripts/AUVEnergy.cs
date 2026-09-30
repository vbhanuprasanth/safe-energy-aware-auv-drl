using UnityEngine;

public class AUVEnergy : MonoBehaviour
{
    [Header("Energy Settings")]
    public float initialEnergy = 100f;

    private float remainingEnergy;

    public float RemainingEnergy
    {
        get { return remainingEnergy; }
    }

    void Awake()
    {
        ResetEnergy();
    }

    public void ResetEnergy()
    {
        remainingEnergy = initialEnergy;
    }

    public float ConsumeEnergy(Vector3 action)
    {
        float actionMagnitude = action.magnitude;

        float energyConsumed = 0.5f * actionMagnitude;

        remainingEnergy -= energyConsumed;

        remainingEnergy = Mathf.Max(0f, remainingEnergy);

        return energyConsumed;
    }

    public bool IsDepleted()
    {
        return remainingEnergy <= 0f;
    }
}